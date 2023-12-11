#include <iostream>
#include <cuda.h>

using namespace std;

#define N_SIZE 4096

__global__ void SigmoidForwardKernel(const float* input, float* output, const int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        output[i] = 1 / (1 + expf(-__ldg(input + i)));
    }
}

void sigmoid(const float* input, float* output, int N) {
    float *input_dev, *output_dev;
    cudaMalloc((void**)&input_dev, sizeof(float) * N);
    cudaMalloc((void**)&output_dev, sizeof(float) * N);
    cudaMemcpy(input_dev, input, sizeof(float) * N, cudaMemcpyHostToDevice);

    int THREADS_PER_BLOCK = 256;
    int BLOCKS_PER_GRID = (N_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 dimGrid(BLOCKS_PER_GRID, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    SigmoidForwardKernel<<<dimGrid, dimBlock>>>(input_dev, output_dev, N);

    cudaMemcpy(output, output_dev, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaFree(input_dev);
    cudaFree(output_dev);
}

int main() {
    float* input = (float*)malloc(N_SIZE * sizeof(float));
    float* output = (float*)malloc(N_SIZE * sizeof(float));

    // srand((unsigned)time(NULL));
    // float rand_min = -10.0, rand_max = 10.0, rand_num = 0.0;
    // for (int i = 0; i < N_SIZE; ++i) {
    //     rand_num = (float)rand()/RAND_MAX;
    //     input[i] = rand_min + rand_num * (rand_max - rand_min);
    // }

    for (int i = 0; i < N_SIZE; ++i) {
       input[i] = 1.0;
    }

    int warmup_steps = 3;
    int steps = 10;
    clock_t start, stop;
    double duration;

    for (int i = 0; i < warmup_steps; ++i) {
        sigmoid(input, output, N_SIZE);
    }
    start=clock();
    for (int i = 0; i < steps; ++i) {
        sigmoid(input, output, N_SIZE);
    }
    stop=clock();
    duration=(double)(stop-start)/(steps*CLOCKS_PER_SEC);
    printf("cudnn time:%f\n", duration);

    printf("Input is \n");
    for (int i = 0; i < N_SIZE; ++i) {
        cout << input[i] << ' ';
    }
    cout << endl;

    printf("Output is \n");
    for (int i = 0; i < N_SIZE; ++i) {
        cout << output[i] << ' ';
    }
    cout << endl;

    free(input);
    free(output);

    return EXIT_SUCCESS;
}