#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <thread>

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

void runRT() {

}

bool cuda_avail() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    return devCount > 0;
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