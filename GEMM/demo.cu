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
#define TILE_WIDTH 16
#define IDX2C(i, j, ld) ((j) * (ld) + (i))

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

__global__  void NaiveSgemm(int m, int n, int k, const float alpha, const float *A, 
    const float *B, const float beta, float *C) {
    // get thread index
    int idx_n = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_m = blockDim.y * blockIdx.y + threadIdx.y;
    // very important: 
    if (idx_n >= n || idx_m >= m) return;

    float sum = 0.0f;
    for(int idx_k = 0; idx_k < k; ++idx_k) {
        float a = A[IDX2C(idx_m, idx_k, m)];
        float b = B[IDX2C(idx_k, idx_n, k)];
        sum += a * b;
    }
    if (beta < 0.0000000001 && beta > -0.0000000001) {
        C[IDX2C(idx_m, idx_n, m)] = sum * alpha;
    } else {
        C[IDX2C(idx_m, idx_n, m)] = sum * alpha + C[IDX2C(idx_m, idx_n, m)] * beta;
    }
}

void gpuNaiveSgemm(int m, int n, int k, const float alpha, 
                   const float *A, const float *B, const float beta, float *C) {
    //malloc on device
    float *devPtrA, *devPtrB, *devPtrC;
    cudaMalloc((void**)&devPtrA, sizeof(float) * m * k);
    cudaMalloc((void**)&devPtrB, sizeof(float) * k * n);
    cudaMalloc((void**)&devPtrC, sizeof(float) * m * n);
    //copy A and B to device
    cudaMemcpy(devPtrA, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrB, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    //use my kernel to compute
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    NaiveSgemm<<<dimGrid, dimBlock>>>(m, n, k, alpha, devPtrA, devPtrB, beta, devPtrC);
    //copy devPtrC to host
    cudaMemcpy(C, devPtrC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    //release memory on device
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
}

__global__  void OptSgemm(int m, int n, int k, const float alpha, const float *A, 
    const float *B, const float beta, float *C) {
    __shared__ float TiledA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float TiledB[TILE_WIDTH][TILE_WIDTH];

    // get thread index
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx_n = blockDim.x * blockIdx.x + tx;
    int idx_m = blockDim.y * blockIdx.y + ty;
    // very important: 
    if (idx_n >= n || idx_m >= m) return;

    float sum = 0;
    for (int idx_tile = 0; idx_tile < k / TILE_WIDTH; ++idx_tile) {
        // A[idx_m][idx_tile * TILE_WIDTH + tx]
        TiledA[ty][tx] = A[IDX2C(idx_m, idx_tile * TILE_WIDTH + tx, m)];
        // B[idx_tile * TILE_WIDTH + ty][idx_n]
        TiledB[ty][tx] = B[IDX2C(idx_tile * TILE_WIDTH + ty, idx_n, k)];
        __syncthreads();
        for (int idx_k = 0; idx_k < TILE_WIDTH; ++idx_k) {
            sum += TiledA[ty][idx_k] * TiledB[idx_k][tx];
        }
        __syncthreads();
    }
    if (beta < 0.0000000001 && beta > -0.0000000001) {
        C[IDX2C(idx_m, idx_n, m)] = sum * alpha;
    } else {
        C[IDX2C(idx_m, idx_n, m)] = sum * alpha + C[IDX2C(idx_m, idx_n, m)] * beta;
    }
}

void gpuOptSgemm(int m, int n, int k, const float alpha, 
                   const float *A, const float *B, const float beta, float *C) {
    //malloc on device
    float *devPtrA, *devPtrB, *devPtrC;
    cudaMalloc((void**)&devPtrA, sizeof(float) * m * k);
    cudaMalloc((void**)&devPtrB, sizeof(float) * k * n);
    cudaMalloc((void**)&devPtrC, sizeof(float) * m * n);
    //copy A and B to device
    cudaMemcpy(devPtrA, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrB, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    //use my kernel to compute
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    OptSgemm<<<dimGrid, dimBlock>>>(m, n, k, alpha, devPtrA, devPtrB, beta, devPtrC);
    //copy devPtrC to host
    cudaMemcpy(C, devPtrC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    //release memory on device
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
}

void gpuBlasSgemm(int m, int n, int k, const float alpha, 
                  const float *A, const float *B, const float beta, float *C) {
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    status = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH); //CUBLAS_TENSOR_OP_MATH
    //malloc on device
    float *devPtrA, *devPtrB, *devPtrC;
    cudaMalloc((void**)&devPtrA, sizeof(float) * m * k);
    cudaMalloc((void**)&devPtrB, sizeof(float) * k * n);
    cudaMalloc((void**)&devPtrC, sizeof(float) * m * n);
    //copy A and B to device
    cublasSetVector (m * k, sizeof(float), A, 1, devPtrA, 1);
    cublasSetVector (k * n, sizeof(float), B, 1, devPtrB, 1);
    //use clublas to compute
    cudaDeviceSynchronize();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devPtrA, m, devPtrB, k, &beta, devPtrC, m);
    cudaDeviceSynchronize();
    //copy devPtrC to host
    cublasGetVector(m * n, sizeof(float), devPtrC, 1, C, 1);
    //release memory on device
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
}

int main() {
    srand((unsigned)time(NULL));
    float rand_min = -10.0, rand_max = 10.0, rand_num = 0.0;

    float* matrix_in1 = (float *)malloc(sizeof(float) * M * K);
    float* matrix_in2 = (float *)malloc(sizeof(float) * K * N);
    float* matrix_out_cpu = (float *)malloc(sizeof(float) * M * N);
    float* matrix_out_gpu_naive = (float *)malloc(sizeof(float) * M * N);
    float* matrix_out_gpu_opt = (float *)malloc(sizeof(float) * M * N);
    float* matrix_out_gpu_blas = (float *)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_in1[i] = rand_min + rand_num * (rand_max - rand_min);
    }
    for (int i = 0; i < K * N; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_in2[i] = rand_min + rand_num * (rand_max - rand_min);
    }

    float a = 2.0, b = 0.0;
    clock_t start, stop;
    double duration;
    int warmup_steps = 3;
    int steps = 10;

    // record cpu 
    for (int i = 0; i < warmup_steps; ++i) {
        cpuSgemm(M, N, K, a, matrix_in1, matrix_in2, b, matrix_out_cpu);
    }
    start=clock();
    for (int i = 0; i < steps; ++i) {
        cpuSgemm(M, N, K, a, matrix_in1, matrix_in2, b, matrix_out_cpu);
    }
    stop=clock();
    duration=(double)(stop-start)/(steps*CLOCKS_PER_SEC);
    printf("cpu with no optimization time:%f\n",duration);


    // record gpu with naive gemm execution time
    for (int i = 0; i < warmup_steps; ++i) {
        gpuNaiveSgemm(M, N, K, a, matrix_in1, matrix_in2, b, matrix_out_gpu_naive);
    }
    start=clock();
    for (int i = 0; i < steps; ++i) {
        gpuNaiveSgemm(M, N, K, a, matrix_in1, matrix_in2, b, matrix_out_gpu_naive);
    }
    stop=clock();
    duration=(double)(stop-start)/(steps*CLOCKS_PER_SEC);
    printf("gpu with no optimization time:%f\n",duration);


    // record gpu with opt gemm execution time
    for (int i = 0; i < warmup_steps; ++i) {
        gpuOptSgemm(M, N, K, a, matrix_in1, matrix_in2, b, matrix_out_gpu_opt);
    }
    start=clock();
    for (int i = 0; i < steps; ++i) {
        gpuOptSgemm(M, N, K, a, matrix_in1, matrix_in2, b, matrix_out_gpu_opt);
    }
    stop=clock();
    duration=(double)(stop-start)/(steps*CLOCKS_PER_SEC);
    printf("gpu with opt time:%f\n",duration);


    // record gpu with cublas execution time
    for (int i = 0; i < warmup_steps; ++i) {
        gpuBlasSgemm(M, N, K, a, matrix_in1, matrix_in2, b, matrix_out_gpu_blas);
    }
    start=clock();
    for (int i = 0; i < steps; ++i) {
        gpuBlasSgemm(M, N, K, a, matrix_in1, matrix_in2, b, matrix_out_gpu_blas);
    }
    stop=clock();
    duration=(double)(stop-start)/(steps*CLOCKS_PER_SEC);
    printf("gpu with cublas time:%f\n",duration);

    // check result                                             
    printf("check naive gemm with cpu gemm\n");
    for (int i = 0; i < M * N; ++i) {
        float error = (matrix_out_gpu_naive[i] - matrix_out_cpu[i]) / matrix_out_cpu[i];
        if (error < -EPSILON || error > EPSILON)
            printf("wrong, %f, %f, %f\n", matrix_out_gpu_naive[i], matrix_out_cpu[i], error);
    }
    printf("right\n");

    printf("check opt gemm with cpu gemm\n");
    for (int i = 0; i < M * N; ++i) {
        float error = (matrix_out_gpu_opt[i] - matrix_out_cpu[i]) / matrix_out_cpu[i];
        if (error < -EPSILON || error > EPSILON)
            printf("wrong, %f, %f, %f\n", matrix_out_gpu_opt[i], matrix_out_cpu[i], error);
    }
    printf("right\n");

    printf("check cublas gemm with cpu gemm\n");
    for (int i = 0; i < M * N; ++i) {
        float error = (matrix_out_gpu_blas[i] - matrix_out_cpu[i]) / matrix_out_cpu[i];
        if (error < -EPSILON || error > EPSILON)
            printf("wrong, %f, %f, %f\n", matrix_out_gpu_blas[i], matrix_out_cpu[i], error);
    }
    printf("right\n");

    //release memory on host
    free(matrix_in1);
    free(matrix_in2);
    free(matrix_out_gpu_naive);
    free(matrix_out_gpu_blas);

    return EXIT_SUCCESS;
}