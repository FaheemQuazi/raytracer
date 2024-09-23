#ifndef _FQRT_TASKS_HPP_
#define _FQRT_TASKS_HPP_

#include <stdint.h>
#include <tira/graphics/camera.h>
#include <glm/vec3.hpp>
#include "fqrt/objects.hpp"

namespace fqrt {
    namespace tasks {
        /**
         * test Intersection of ray with sphere
         *   c: camera position
         *   d: direction vector
         *   s: sphere position
         *   r: sphere radius
         */
        bool testIntersectSphere(glm::vec3 c, glm::vec3 d, glm::vec3 s, float r) {
            glm::vec3 scdist = s - c;
            float dc = glm::dot(scdist, scdist) - std::pow(r, 2);
            float db = 2 * (glm::dot(d, scdist));
            float da = glm::dot(d, d);

            float disc = std::pow(db, 2) - (4.0 * da * dc);
            return disc >= 0;
        }
    }
}

#endif