// nvcc -I /nfs/site/home/zhoujia2/chenfeiz/cutlass/include -lcublas cutlass.cu -o cutlass 

#include <iostream>                                           
#include "cutlass/gemm/device/gemm.h"       
 
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;
 
using CutlassGemm = cutlass::gemm::device::Gemm<float,    
                                                RowMajor,
                                                float,
                                                RowMajor,
                                                float,
                                                RowMajor>;
 
void generate_tensor_2D(float *ptr, int i_M, int i_N) {
    for(int i = 0; i < i_M; i++){
        for(int j = 0; j < i_N; j++){
            *(ptr + i*i_N + j ) = 1.0;
        }
    }
}
 
int main(int argc, const char *arg[]) {
 
    int M = 3840;           //M
    int N = 4096;           //N
    int K = 4096;           //K
 
    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = N;
 
    float alpha = 1.0;      //alpha
    float beta = 1.0;       //beta
 
    float *A;    
    float *B;
    float *C;
    float *D;
 
    size_t A_mem_size = sizeof(float) * M * K; //memory size of matrix A = M * K * sizeof(float)
    size_t B_mem_size = sizeof(float) * K * N; //memory size of matrix B = K * N * sizeof(float)
    size_t C_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
    size_t D_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
 
    A = (float*)malloc(A_mem_size);
    B = (float*)malloc(B_mem_size);
    C = (float*)malloc(C_mem_size);
    D = (float*)malloc(D_mem_size);
 
    generate_tensor_2D(A, M, K);
    generate_tensor_2D(B, K, N); 
    generate_tensor_2D(C, M, N);
 
    float *d_A;
    float *d_B;
    float *d_C;
    float *d_D;
 
    cudaMalloc((void**)&d_A, A_mem_size);
    cudaMalloc((void**)&d_B, B_mem_size);
    cudaMalloc((void**)&d_C, C_mem_size);
    cudaMalloc((void**)&d_D, D_mem_size);
 
    cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, C_mem_size, cudaMemcpyHostToDevice);
 
    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
                                {d_A, lda},     // source matrix A
                                {d_B, ldb},     // source matrix B
                                {d_C, ldc},     // source matrix C
                                {d_D, ldd},     // destination matrix D
                                {alpha, beta}); // alpha & beta
    gemm_operator(args);
 
    cudaMemcpy(D, d_D, D_mem_size, cudaMemcpyDeviceToHost);
    std::cout << D[0] << std::endl;
    std::cout << D[M * N - 1] << std::endl;
    return 0;
} 