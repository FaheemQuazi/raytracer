#ifndef RT_CUH__
#define RT_CUH__

#include "fqrt/scene_gpu.hpp"

void cuda_hello();
bool cuda_avail();
void runRT(fqrt::scene::sceneDataGpu_t* sd);

#endif