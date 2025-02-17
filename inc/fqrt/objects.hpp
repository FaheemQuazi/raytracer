#ifndef _FQRT_OBJECTS_HPP_
#define _FQRT_OBJECTS_HPP_

#include <stdint.h>
#include <glm/vec3.hpp>

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
}

#endif