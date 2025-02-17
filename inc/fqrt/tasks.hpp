#ifndef CUDA_CALLABLE

//define the CUDA_CALLABLE macro (will prefix all members)
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__ inline
#else
#define CUDA_CALLABLE
#endif

#ifdef __CUDACC__
#define CUDA_UNCALLABLE __host__ inline
#else
#define CUDA_UNCALLABLE
#endif

#endif

#ifndef _FQRT_TASKS_HPP_
#define _FQRT_TASKS_HPP_

#include <stdint.h>
#include <glm/vec3.hpp>
#include "fqrt/objects.hpp"

namespace fqrt {
    namespace tasks {
        typedef struct hitTestResult_tag {
            bool valid;
            glm::vec3 pos;
            glm::vec3 nor;
            float t;
        } hitTestResult;
        /**
         * build direction ray between two points
         *   a: start point
         *   b: end point
         */
        CUDA_CALLABLE glm::vec3 buildDirRay(glm::vec3 a, glm::vec3 b) {
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
        CUDA_CALLABLE void traceIntersectSphere(glm::vec3 p, glm::vec3 d, glm::vec3 s, float r, hitTestResult* out) {
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
            out->t = th;
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
        CUDA_CALLABLE void traceIntersectSphere(glm::vec3 p, glm::vec3 d, fqrt::objects::sphere S, hitTestResult* out) {
            traceIntersectSphere(p, d, S.pos, S.r, out);
        }
        
        CUDA_CALLABLE void traceIntersectPlane(glm::vec3 p, glm::vec3 d, fqrt::objects::plane P, hitTestResult* out) {
            out->valid = false;

            float ndd = glm::dot(P.norm, d);
            if (ndd == 0) return; // undefined (parallel)
            glm::vec3 v_cp = P.pos - p;
            float th = glm::dot(v_cp, P.norm) / ndd;
            if (th <= 0) return; // behind/on camera

            out->pos = p + (th * d);
            out->nor = P.norm;
            out->t = th;
            out->valid = true;
        }

        CUDA_CALLABLE void traceIntersectTriangle(glm::vec3 p, glm::vec3 d, tira::triangle T, hitTestResult* out) {
            out->valid = false;
            
            // make a quick plane to run initial hit-test
            hitTestResult hrt_PLT;
            fqrt::objects::plane PLT = {
                .pos = T.v[0],
                .norm = T.n
            };
            traceIntersectPlane(p, d, PLT, &hrt_PLT);
            if (!hrt_PLT.valid) return;
            // test if hit inside triangle
            bool leftAB = glm::dot(glm::cross((T.v[1] - T.v[0]), (hrt_PLT.pos - T.v[0])), T.n) >= 0;
            bool leftCB = glm::dot(glm::cross((T.v[2] - T.v[1]), (hrt_PLT.pos - T.v[1])), T.n) >= 0;
            bool leftAC = glm::dot(glm::cross((T.v[0] - T.v[2]), (hrt_PLT.pos - T.v[2])), T.n) >= 0;
            if (!(leftAB && leftCB && leftAC)) return;

            out->pos = hrt_PLT.pos;
            out->nor = T.n;
            out->t = hrt_PLT.t;
            out->valid = true;
        }

        /**
         * calc Illumination value for basic object
         *   l: light position
         *   p: object position
         *   n: object normal
         */
        CUDA_CALLABLE float calcLightingAtPos(glm::vec3 l, glm::vec3 p, glm::vec3 n) {
            return glm::max(glm::dot(n ,(l - p) / glm::length(l - p)),0.0f);
        }
    }
}

#endif