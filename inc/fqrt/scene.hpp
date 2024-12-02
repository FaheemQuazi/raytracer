#ifndef _FQRT_SCENE_HPP_
#define _FQRT_SCENE_HPP_

#include <stdint.h>
#include <glm/vec3.hpp>
#include "fqrt/objects.hpp"
#include "tira/graphics/camera.h"
#include "tira/graphics/shapes/simplemesh.h"
#include "tira/image.h"

namespace fqrt {
    namespace scene {
        typedef struct sceneData_tag {
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
            int meshCount;
            tira::simplemesh* meshes;

            tira::image<unsigned char> img;
        } sceneData_t;
    }
}

#endif