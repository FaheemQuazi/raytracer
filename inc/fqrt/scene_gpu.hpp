#ifndef _FQRT_SCENE_GPU_HPP_
#define _FQRT_SCENE_GPU_HPP_

#include <stdint.h>
#include <glm/vec3.hpp>
#include "tira/graphics/camera.h"
#include "fqrt/objects.hpp"

#define IMG_DATA(c) glm::vec<(c), unsigned char>
namespace fqrt {
    namespace scene {
        typedef struct sceneDataGpu_tag {
            tira::camera cam;
            glm::vec3 cam_bg;

            float dW;
            float dH;
            float dC;

            int sphereCount;
            fqrt::objects::sphere *spheres;
            int planeCount;
            fqrt::objects::plane *planes;
            int lightCount;
            fqrt::objects::light *lights;

            uint8_t* img;
        } sceneDataGpu_t;
    }
}

#endif