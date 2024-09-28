#ifndef _FQRT_TASKS_HPP_
#define _FQRT_TASKS_HPP_

#include <stdint.h>
#include <tira/graphics/camera.h>
#include <glm/vec3.hpp>
#include "fqrt/objects.hpp"

namespace fqrt {
    namespace tasks {
        typedef struct hitTestResult_tag {
            bool valid;
            glm::vec3 pos;
            glm::vec3 nor;
        } hitTestResult;
        /**
         * build direction ray between two points
         *   a: start point
         *   b: end point
         */
        glm::vec3 buildDirRay(glm::vec3 a, glm::vec3 b) {
            return glm::normalize(b - a);
        }
        /**
         * Trace Intersection of ray with sphere
         *   p: ray position
         *   d: ray unit direction vector
         *   s: sphere position (center)
         *   r: sphere radius
         *   out: hit result data
         */
        void traceIntersectSphere(glm::vec3 p, glm::vec3 d, glm::vec3 s, float r, hitTestResult* out) {
            out->valid = false;

            glm::vec3 v_cs = s - p; // vec from pos to sphere
            float sq_cs = glm::dot(v_cs, v_cs); // squared distance
            float sq_r = glm::pow(r, 2); // squared radius
            if (sq_cs < sq_r) return; // inside sphere
            float s_proj = glm::dot(v_cs, d); // ray pos-sphere vec projected in ray direction
            if (s_proj < 0) return; // sphere behind ray
            float sq_m = sq_cs - glm::pow(s_proj, 2);
            if (sq_r < sq_m) return; // ray misses
            // at this point, confirmed ray hit
            float th = s_proj - sqrt(sq_r - sq_m); // t-scalar

            // hit data output
            glm::vec3 oPos = p + (th * d);
            out->pos = oPos;
            out->nor = glm::normalize(oPos - s);
            out->valid = true;
        }
        /**
         * Trace Intersection of ray with sphere
         *   p: initial position
         *   d: unit direction vector
         *   S: sphere object
         *   out: hit result data
         */
        void traceIntersectSphere(glm::vec3 p, glm::vec3 d, fqrt::objects::sphere S, hitTestResult* out) {
            traceIntersectSphere(p, d, S.pos, S.r, out);
        }
        /**
         * calc Illumination value for basic object
         *   l: light position
         *   p: object position
         *   n: object normal
         */
        float calcLightingAtPos(glm::vec3 l, glm::vec3 p, glm::vec3 n) {
            return glm::max(glm::dot(n ,(l - p) / glm::length(l - p)),0.0f);
        }
    }
}

#endif