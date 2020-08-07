/*
nvcc hello_parallel.cu -o para
*/

#include <stdio.h>

using namespace std;

int dev_x=1024;
int dev_y=1024;

double *num1_d, *num2_d;

double num1[1048576];
double num2[1048576];


__global__ void hello(double* num1, double* num2){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i%200000==0){
        num1[i] = num1[i] + num2[i];
    }else{
        num1[i] = num1[i] - num2[i];
    }
}

int main() {
    int blocksize;

    blocksize = dev_y;
    dim3 block (blocksize, 1, 1);
    dim3 grid  ((dev_x*dev_y)/block.x, 1, 1);

    size_t data_size = sizeof(double)*dev_x*dev_y;
    cudaMalloc((void**) &num1_d, data_size);
    cudaMalloc((void**) &num2_d, data_size);
    //num1_h = (double*) malloc(data_size);
    //num2_h = (double*) malloc(data_size);

    for(int i=0; i<1048576; i++){
        num1[i] = i;
        num2[i] = i;
        //num1_h[i] = num1[i];
        //num2_h[i] = num2[i];
    }

    cudaMemcpy(num1_d, num1, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(num2_d, num2, data_size, cudaMemcpyHostToDevice);

    hello<<< grid, block >>>(num1_d, num2_d);
    cudaDeviceSynchronize();

    cudaMemcpy(num1, num1_d, data_size, cudaMemcpyDeviceToHost);

    for(int i=0; i<1048576; i=i+100000)printf("No.%d value : %g\n", i, num1[i]);

    cudaFree(num1_d);
    cudaFree(num2_d);
    //free(num1_h);
    //free(num2_h);

    return 0;
}
