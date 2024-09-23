#ifndef _FQRT_OBJECTS_HPP_
#define _FQRT_OBJECTS_HPP_

#include <stdint.h>
#include <glm/vec3.hpp>

namespace fqrt {
    namespace objects {
        typedef struct sphere_tag {
            float r;
            glm::vec3 pos;
            glm::vec3 color;
        } sphere;

        typedef struct light_tag {
            glm::vec3 pos;
            glm::vec3 color;
        } light;
    }
}

#endif