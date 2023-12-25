// nvcc tensor_core.cu -o tensor_core -arch=compute_86 -code=sm_86 -lcublas --ptxas-options=-v

#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
using namespace std;

__global__ void demo(half* a, half* b, half* c, const int M, const int N) {

    const int BM = 32;
    const int BN = 64;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int fy = (wid % 2 == 0) ? 0 : 16;
    int fx = (wid / 2) * 16;

    int offset = (by * BM + fy) * N + bx * BN + fx;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;
    wmma::fill_fragment(frag_c, 0.0);
    wmma::load_matrix_sync(frag_a, a + offset, N);
    wmma::load_matrix_sync(frag_b, b + offset, N);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    wmma::store_matrix_sync(c + offset, frag_c, N, wmma::mem_row_major);
}

void run_demo(half* a, half* b, half* c, int M, int N) {
    half *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, sizeof(half) * M * N);
    cudaMalloc((void**)&b_d, sizeof(half) * M * N);
    cudaMalloc((void**)&c_d, sizeof(half) * M * N);
    cudaMemcpy(a_d, a, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, sizeof(half) * M * N, cudaMemcpyHostToDevice);

    const int BM = 32, BN = 64;
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
    int M = 128;
    int N = 128;

    half *a, *b, *c;
    a = (half*)malloc(M * N * sizeof(half));
    b = (half*)malloc(M * N * sizeof(half));
    c = (half*)malloc(M * N * sizeof(half));
    for (int i = 0; i < M * N; ++i) {
        a[i] = 1.0;
        b[i] = 1.0;
        c[i] = 2.0;
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