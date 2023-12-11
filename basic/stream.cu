#include <iostream>
#include <cuda.h>

using namespace std;

#define N_SIZE 20 * (1 << 20)

__global__ void SAXYKernel(const float* A, float* B, const float alpha, const int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] = alpha * A[i] + B[i];
    }
}

__global__ void FMAKernel(const float* A, const float* B, float* C, const int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] * B[i] + C[i];
    }
}

void test(const float* A, const float* B, float* C, int N) {
    float *A_dev, *B_dev, *C_dev;
    cudaMalloc((void**)&A_dev, sizeof(float) * N);
    cudaMalloc((void**)&B_dev, sizeof(float) * N);
    cudaMalloc((void**)&C_dev, sizeof(float) * N);
    cudaMemcpy(A_dev, A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(C_dev, C, sizeof(float) * N, cudaMemcpyHostToDevice);

    int THREADS_PER_BLOCK = 256;
    int BLOCKS_PER_GRID = (N_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 dimGrid(BLOCKS_PER_GRID, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    int warmup_steps = 3;
    int steps = 10;
    float milliseconds = 0;
    int total_bytes;
    float elapsed_time;
    float gmem_bandwidth;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_steps; ++i) {
        SAXYKernel<<<dimGrid, dimBlock>>>(A_dev, B_dev, 1.0, N);
    }    
    cudaEventRecord(start);
    for (int i = 0; i < steps; ++i) {
        SAXYKernel<<<dimGrid, dimBlock>>>(A_dev, B_dev, 1.0, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_bytes = N * 3 * 4;
    elapsed_time = milliseconds / steps;
    gmem_bandwidth = total_bytes / elapsed_time / 1e6;
    printf("SAXY Bandwidth (GB/s): %f \n", gmem_bandwidth);

    for (int i = 0; i < warmup_steps; ++i) {
        FMAKernel<<<dimGrid, dimBlock>>>(A_dev, B_dev, C_dev, N);
    }    
    cudaEventRecord(start);
    for (int i = 0; i < steps; ++i) {
        FMAKernel<<<dimGrid, dimBlock>>>(A_dev, B_dev, C_dev, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_bytes = N * 4 * 4;
    elapsed_time = milliseconds / steps;
    gmem_bandwidth = total_bytes / elapsed_time / 1e6;
    printf("FMA Bandwidth (GB/s): %f \n", gmem_bandwidth);

    // cudaMemcpy(C, C_dev, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
}

int main() {
    float* A = (float*)malloc(N_SIZE * sizeof(float));
    float* B = (float*)malloc(N_SIZE * sizeof(float));
    float* C = (float*)malloc(N_SIZE * sizeof(float));

    for (int i = 0; i < N_SIZE; ++i) {
       A[i] = 1.0;
       B[i] = 1.0;
       C[i] = 1.0;
    }

    test(A, B, C, N_SIZE);

    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}