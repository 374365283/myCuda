// nvcc gemm.cu -o gemm -arch=compute_86 -code=sm_86 --ptxas-options=-v

#include <stdint.h>
#include <algorithm>
#include <iostream>

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
// #include "cublas_v2.h"

using namespace std;

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

__global__ void mmaNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2) {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}

void run_mmaNaive(half* a, half* b, half* c, size_t M, size_t N, size_t K) {
    half *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, sizeof(half) * M * K);
    cudaMalloc((void**)&b_d, sizeof(half) * K * N);
    cudaMalloc((void**)&c_d, sizeof(half) * M * N);
    cudaMemcpy(a_d, a, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(half) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, sizeof(half) * M * N, cudaMemcpyHostToDevice);

    dim3 blockDim(WARP_SIZE);
    dim3 gridDim(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
    mmaNaiveKernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

    cudaMemcpy(c, c_d, sizeof(half) * M * N, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    size_t M = 128;
    size_t N = 128;
    size_t K = 128;

    half *a, *b, *c;
    a = (half*)malloc(M * N * sizeof(half));
    b = (half*)malloc(M * N * sizeof(half));
    c = (half*)malloc(M * N * sizeof(half));
    for (int i = 0; i < M * N; ++i) {
        a[i] = 0.1;
        b[i] = 0.1;
        c[i] = 1.0;
    }

    run_mmaNaive(a, b, c, M, N, K);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << (float)c[i * N + j] << ' ';
        }
        cout << endl;
    }

    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}


