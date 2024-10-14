#ifndef _FQRT_OBJECTS_HPP_
#define _FQRT_OBJECTS_HPP_

#include <stdint.h>
#include <glm/vec3.hpp>
#include "tira/graphics/camera.h"
#include "tira/graphics/shapes/simplemesh.h"
#include "tira/image.h"

#define SURF_OFFSET_SPHERE -0.01f
#define SURF_OFFSET_PLANE   0.01f

namespace fqrt {
    namespace objects {
        typedef struct sphere_tag {
            float r;
            glm::vec3 pos;
            glm::vec3 color;
        } sphere;
        typedef struct plane_tag {
            glm::vec3 pos;
            glm::vec3 norm;
            glm::vec3 color;
        } plane;

        typedef struct light_tag {
            glm::vec3 pos;
            glm::vec3 color;
        } light;
    }
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