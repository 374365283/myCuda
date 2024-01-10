// nvcc cublas.cu -lcublas -o cublas 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define M 128
#define N 128
#define K 128
#define EPSILON 0.01
#define IDX2C(i, j, ld) ((j) * (ld) + (i))
#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

void cpuSgemm(int m, int n, int k, const float alpha, const float *A, const float *B,
              const float beta, float *C) {
    for (int idx_m = 0; idx_m < m; ++idx_m) {
        for (int idx_n = 0; idx_n < n; ++idx_n) {
            float sum = 0.0;
            for (int idx_k = 0; idx_k < k; ++idx_k) {
                sum += A[IDX2C(idx_m, idx_k, m)] * B[IDX2C(idx_k, idx_n, k)];
            }
            C[IDX2C(idx_m, idx_n, m)] = alpha * sum + beta * C[IDX2C(idx_m, idx_n, m)];
        }
    }
}

int main() {
    srand((unsigned)time(NULL));
    float rand_min = -10.0, rand_max = 10.0, rand_num = 0.0;

    float* matrix_in1 = (float *)malloc(sizeof(float) * M * K);
    float* matrix_in2 = (float *)malloc(sizeof(float) * K * N);
    float* matrix_out_cpu = (float *)malloc(sizeof(float) * M * N);
    float* matrix_out_gpu_cublas = (float *)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_in1[i] = rand_min + rand_num * (rand_max - rand_min);
    }
    for (int i = 0; i < K * N; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_in2[i] = rand_min + rand_num * (rand_max - rand_min);
    }
    for (int i = 0; i < M * N; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_out_cpu[i] = rand_min + rand_num * (rand_max - rand_min);
    }
    for (int i = 0; i < M * N; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_out_gpu_cublas[i] = rand_min + rand_num * (rand_max - rand_min);
    }

    const float alpha = 2.0, beta = 0.0;
    int warmup_steps = 3;
    int steps = 10;

    // run cpu gemm.
    printf("Start running cpu gemm.\n");
    cpuSgemm(M, N, K, alpha, matrix_in1, matrix_in2, beta, matrix_out_cpu);
    printf("Finish running cpu gemm.\n");

    // run cublas gemm.
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    status = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH); //CUBLAS_TENSOR_OP_MATH

    float *devPtrA, *devPtrB, *devPtrC;
    cudaMalloc((void**)&devPtrA, sizeof(float) * M * K);
    cudaMalloc((void**)&devPtrB, sizeof(float) * K * N);
    cudaMalloc((void**)&devPtrC, sizeof(float) * M * N);
    checkCudaErrors(cudaMemcpy(devPtrA, matrix_in1, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devPtrB, matrix_in2, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // cudaDeviceSynchronize();
    printf("Start running cublas gemm warmup.\n");
    for (int i = 0; i < warmup_steps; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, devPtrA, M, devPtrB, K, &beta, devPtrC, M);
    }
    printf("Finish running cublas gemm warmup.\n");

    cudaEvent_t start, stop;
    float msecTotal;
    double flopsPerMatrixMul;
    float msecPerMatrixMul;
    double gigaFlops;
    printf("Start running cublas gemm performance.\n");
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < steps; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, devPtrA, M, devPtrB, K, &beta, devPtrC, M);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("Finish running cublas gemm performance.\n");
    // cudaDeviceSynchronize();

    flopsPerMatrixMul = 2.0 * M * N * K;
    msecPerMatrixMul = msecTotal / steps;
    gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n", gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    checkCudaErrors(cudaMemcpy(matrix_out_gpu_cublas, devPtrC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
                                         
    printf("Check cublas gemm with cpu gemm. \n");
    for (int i = 0; i < M * N; ++i) {
        float error = (matrix_out_gpu_cublas[i] - matrix_out_cpu[i]) / matrix_out_cpu[i];
        if (error < -EPSILON || error > EPSILON)
            printf("wrong, %f, %f, %f\n", matrix_out_gpu_cublas[i], matrix_out_cpu[i], error);
    }
    printf("right\n");

    // release memory on host
    free(matrix_in1);
    free(matrix_in2);
    free(matrix_out_cpu);
    free(matrix_out_gpu_cublas);

    return EXIT_SUCCESS;
}