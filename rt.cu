#include <stdio.h>
#include "rt.cuh"

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

void cuda_hello()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("%d: %s:%d.%d\n", i, props.name, props.major, props.minor);
    }
    helloCUDA<<<3, 1>>>();
    cudaDeviceSynchronize();
    return;
}