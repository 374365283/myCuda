#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

using namespace std;

__global__ void demo(half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    int tx = (tid % 2 == 0) ? 0 : 128;
    int ty = tid / 2;
    int thread_start_offset = (by * BM + ty) * N + bx * BN + tx;
    for (int i = 0; i < 128; ++i) {
        c[thread_start_offset + i] = a[thread_start_offset + i] + b[thread_start_offset + i] + c[thread_start_offset + i];
    }
}

void run_demo(half* a, half* b, half* c, int M, int N) {
    half *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, sizeof(half) * M * N);
    cudaMalloc((void**)&b_d, sizeof(half) * M * N);
    cudaMalloc((void**)&c_d, sizeof(half) * M * N);
    cudaMemcpy(a_d, a, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, sizeof(half) * M * N, cudaMemcpyHostToDevice);

    const int BM = 128, BN = 256;
    dim3 blockDim(256);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    dim3 gridDim(BX, BY);
    demo<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N);

    cudaMemcpy(c, c_d, sizeof(half) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    int M = 512;
    int N = 512;

    half *a, *b, *c;
    a = (half*)malloc(M * N * sizeof(half));
    b = (half*)malloc(M * N * sizeof(half));
    c = (half*)malloc(M * N * sizeof(half));
    for (int i = 0; i < M * N; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }

    run_demo(a, b, c, M, N);

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