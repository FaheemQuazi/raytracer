#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include "tira/graphics/shapes/simplemesh.h"
#include "fqrt/scene_gpu.hpp"
#include "fqrt/tasks.hpp"
#include "fqrt/util.hpp"

#define CUDA_VALID(n) {cudaError_t x = (n); if ((x) > 0) { printf("CUDA ERROR @ %s:%d - %d\n", __FILE__, __LINE__, (x)); exit(1);}}

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

__device__ void setPixVal(fqrt::scene::sceneDataGpu_t* sceneGPU, uint x, uint y, uint8_t r, uint8_t g, uint8_t b) {
    sceneGPU->img[(y * (int)sceneGPU->dC * (int)(sceneGPU->dH)) + (x * (int)sceneGPU->dC)] = r;
    sceneGPU->img[(y * (int)sceneGPU->dC * (int)(sceneGPU->dH)) + (x * (int)sceneGPU->dC) + 1] = g;
    sceneGPU->img[(y * (int)sceneGPU->dC * (int)(sceneGPU->dH)) + (x * (int)sceneGPU->dC) + 2] = b;
}

__global__ void renderPixel(fqrt::scene::sceneDataGpu_t sd) {
    uint y = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint x = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= sd.dW || y >= sd.dH) return;
   
    // set background color
    setPixVal(&sd, x, y, (uint8_t)(sd.cam_bg.r * 255), (uint8_t)(sd.cam_bg.g * 255), (uint8_t)(sd.cam_bg.b * 255));

    // Image Plane Coordinates
    float ipX = (x - (sd.dW / 2.0)) / sd.dW; /* [-0.5, 0.5]*/
    float ipY = (y - (sd.dH / 2.0)) / sd.dH; /* [-0.5, 0.5]*/

    // Hit test
    glm::vec3 cR = sd.cam.ray(ipX, ipY);
    fqrt::tasks::hitTestResult hrt = {
        .valid = false
    };
    glm::vec3 cHitObjCol(0);

    for (int csph = 0; csph < sd.sphereCount; csph++) { // loop each sphere
        fqrt::tasks::hitTestResult chrt;
        fqrt::tasks::traceIntersectSphere(sd.cam.position(), cR, sd.spheres[csph], &chrt);
        if (chrt.valid) {
            if (hrt.valid && hrt.t > chrt.t) { // found closer sphere
                cHitObjCol = sd.spheres[csph].color;
                hrt = chrt;
            } else if (!hrt.valid) { // havent found anything yet
                cHitObjCol = sd.spheres[csph].color;
                hrt = chrt;
            }
        }
    }

    for (int cpl = 0; cpl < sd.planeCount; cpl++) { // loop each plane
        fqrt::tasks::hitTestResult chrt;
        fqrt::tasks::traceIntersectPlane(sd.cam.position(), cR, sd.planes[cpl], &chrt);
        if (chrt.valid) {
            if (hrt.valid && hrt.t > chrt.t) { // found closer plane
                cHitObjCol = sd.planes[cpl].color;
                hrt = chrt;
            } else if (!hrt.valid) { // haven't found anything yet
                cHitObjCol = sd.planes[cpl].color;
                hrt = chrt;
            }
        }
    }

    if (hrt.valid) { // we got an object here
        glm::vec3 pxCol(0);
        for (int lg = 0; lg < sd.lightCount; lg++) { // loop each light
            bool lightObstruct = false;
            for (int b = 0; b < sd.sphereCount; b++) {
                fqrt::tasks::hitTestResult hrt_l;
                fqrt::tasks::traceIntersectSphere(hrt.pos + SURF_OFFSET_SPHERE*hrt.nor, // move up a hair off the surface
                    fqrt::tasks::buildDirRay(hrt.pos + SURF_OFFSET_SPHERE*hrt.nor, sd.lights[lg].pos),
                    sd.spheres[b], &hrt_l);
                if (hrt_l.valid) { // we are obstructed if this hits
                    lightObstruct = true;
                    break;
                }
            }
            if (!lightObstruct && sd.planeCount > 0) { // if not obstructed check planes
                for (int b = 0; b < sd.planeCount; b++) {
                    fqrt::tasks::hitTestResult hrt_l;
                    fqrt::tasks::traceIntersectPlane(hrt.pos + SURF_OFFSET_PLANE*hrt.nor, // move up a hair off the surface
                        fqrt::tasks::buildDirRay(hrt.pos + SURF_OFFSET_PLANE*hrt.nor, sd.lights[lg].pos),
                        sd.planes[b], &hrt_l);
                    if (hrt_l.valid) { // we are obstructed if this hits
                        lightObstruct = true;
                        break;
                    }
                }
            }
            if (!lightObstruct) { // if not obstructed, incorporate this light
                float intensity = fqrt::tasks::calcLightingAtPos(sd.lights[lg].pos, 
                    hrt.pos, hrt.nor);
                pxCol = pxCol + intensity * sd.lights[lg].color * cHitObjCol;
            }
        }

        setPixVal(&sd, x, y, (uint8_t)(pxCol.r * 255), (uint8_t)(pxCol.g * 255), (uint8_t)(pxCol.b * 255));
    }
}


__host__ void runRT(fqrt::scene::sceneDataGpu_t* sceneCPU) {
    // pass in gpu friendly scene
    fqrt::scene::sceneDataGpu_t sceneGPU;
    sceneGPU.cam = sceneCPU->cam;
    sceneGPU.cam_bg = sceneCPU->cam_bg;
    sceneGPU.dW = sceneCPU->dW;
    sceneGPU.dH = sceneCPU->dH;
    sceneGPU.dC = sceneCPU->dC;
    sceneGPU.sphereCount = sceneCPU->sphereCount;
    CUDA_VALID(cudaMalloc(&sceneGPU.spheres, sizeof(fqrt::objects::sphere) * sceneGPU.sphereCount));
    CUDA_VALID(cudaMemcpy(sceneGPU.spheres, sceneCPU->spheres, sizeof(fqrt::objects::sphere) * sceneGPU.sphereCount, cudaMemcpyHostToDevice));
    sceneGPU.planeCount = sceneCPU->planeCount;
    CUDA_VALID(cudaMalloc(&sceneGPU.planes, sizeof(fqrt::objects::plane) * sceneGPU.planeCount));
    CUDA_VALID(cudaMemcpy(sceneGPU.planes, sceneCPU->planes, sizeof(fqrt::objects::plane) * sceneGPU.planeCount, cudaMemcpyHostToDevice));
    sceneGPU.lightCount = sceneCPU->lightCount;
    CUDA_VALID(cudaMalloc(&sceneGPU.lights, sizeof(fqrt::objects::light) * sceneGPU.lightCount));
    CUDA_VALID(cudaMemcpy(sceneGPU.lights, sceneCPU->lights, sizeof(fqrt::objects::light) * sceneGPU.lightCount, cudaMemcpyHostToDevice));
    CUDA_VALID(cudaMalloc(&sceneGPU.img, sizeof(uint8_t) * sceneCPU->dW * sceneCPU->dH * sceneCPU->dC));
    CUDA_VALID(cudaDeviceSynchronize());

    // Configure Threads
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(sceneGPU.dW/threadsPerBlock.x, sceneGPU.dH/threadsPerBlock.y);
    printf("Using %dx%d blocks and %dx%d TPB\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y );
    renderPixel<<<numBlocks,threadsPerBlock>>>(sceneGPU);
    CUDA_VALID(cudaDeviceSynchronize());

    // copy resulting image back
    CUDA_VALID(cudaMemcpy(sceneCPU->img, sceneGPU.img, sizeof(uint8_t) * sceneCPU->dW * sceneCPU->dH * sceneCPU->dC, cudaMemcpyDeviceToHost));
    CUDA_VALID(cudaDeviceSynchronize());
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
    // helloCUDA<<<3, 1>>>();
    cudaDeviceSynchronize();
    return;
}