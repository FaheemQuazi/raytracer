#include <iostream>
#include <chrono>
#include "tira/parser.h"
#include "tira/graphics/camera.h"
#include "tira/image.h"
#include "fqrt/objects.hpp"
#include "fqrt/tasks.hpp"

using namespace std::chrono;
#define TIME_NOW high_resolution_clock::now()
#define TIME_DURATION(s, f) duration_cast<duration<double>>(f - s).count()

 // load scene
#ifndef SCENE_FILE
#define SCENE_FILE argv[1]
#endif

int main(int argc, char* argv[])
{
    std::cout << "Hello FQRT\n";
    if (argc < 2) {
        std::cerr << "No scene file specified!\n";
        return 1;
    }
    std::cout << "Scene File: " << SCENE_FILE << "\n";

    // timing functions
    high_resolution_clock::time_point tm_start = TIME_NOW;

    tira::parser scene(SCENE_FILE);

    // scene objects
    tira::camera cam;
    glm::vec3 cam_bg;
    fqrt::objects::sphere *spheres;
    fqrt::objects::light *lights;

    // output image
    float img_W = scene.get<float>("resolution", 0);
    float img_H = scene.get<float>("resolution", 1);
    float img_C = 3;
    tira::image<unsigned char> img(img_W, img_H, img_C);

    // configure camera
    cam.position(scene.get<float>("camera_position", 0), scene.get<float>("camera_position", 1), scene.get<float>("camera_position", 2));
    cam.lookat(scene.get<float>("camera_look", 0), scene.get<float>("camera_look", 1), scene.get<float>("camera_look", 2));
    cam.up(glm::vec3(scene.get<float>("camera_up", 0), scene.get<float>("camera_up", 1)*-1, scene.get<float>("camera_up", 2))); // why -1?
    cam.fov(scene.get<float>("camera_fov", 0));
    cam_bg = glm::vec3(scene.get<float>("background", 0), scene.get<float>("background", 1), scene.get<float>("background", 2));

    // allocate spheres
    int sphereCount = scene.count("sphere");
    if (sphereCount > 0) {
        spheres = (fqrt::objects::sphere *)malloc(sizeof(fqrt::objects::sphere) * sphereCount);
        for (int i = 0; i < sphereCount; i++) {
            spheres[i].r = scene.get<float>("sphere", i, 0);
            spheres[i].pos.x = scene.get<float>("sphere", i, 1);
            spheres[i].pos.y = scene.get<float>("sphere", i, 2);
            spheres[i].pos.z = scene.get<float>("sphere", i, 3);
            spheres[i].color.r = scene.get<float>("sphere", i, 4);
            spheres[i].color.g = scene.get<float>("sphere", i, 5);
            spheres[i].color.b = scene.get<float>("sphere", i, 6);
        }
    } else {
        printf("Error: No Spheres in scene!\n");
        return 1;
    }

    // allocate lights
    int lightCount = scene.count("light");
    if (lightCount > 0) {
        lights = (fqrt::objects::light *)malloc(sizeof(fqrt::objects::light) * lightCount);
        for (int i = 0; i < lightCount; i++) {
            lights[i].pos.x = scene.get<float>("light", i, 0);
            lights[i].pos.y = scene.get<float>("light", i, 1);
            lights[i].pos.z = scene.get<float>("light", i, 2);
            lights[i].color.r = scene.get<float>("light", i, 3);
            lights[i].color.g = scene.get<float>("light", i, 4);
            lights[i].color.b = scene.get<float>("light", i, 5);
        }
    } else {
        printf("Error: No Lights in scene!\n");
        return 1;
    }
    
    high_resolution_clock::time_point tm_loaded = TIME_NOW;
    // print what we've loaded so far
    std::cout << "Loaded:\n" \
        << " - Camera: " << cam \
        << " - Render Image: " << img_W << " X " << img_H << std::endl\
        << " - Number of Spheres: " << sphereCount << std::endl \
        << " - Number of Lights: " << lightCount << std::endl;

    // Time tracking stuff
    double* tm_hitTests = (double*)malloc(sizeof(double) * img_H * img_W);
    double* tm_lightTests = (double*)malloc(sizeof(double) * img_H * img_W);

    // Begin Render Pass
    std::cout << "begin render...";
    high_resolution_clock::time_point tm_renderBegin = TIME_NOW;
    high_resolution_clock::time_point tm_rSt;
    high_resolution_clock::time_point tm_rEn;
    int p = 0;
    for (int y = 0; y < img_H; y++) {
        for (int x = 0; x < img_W; x++) {
            // set background color
            img(x, y, 0) = cam_bg.r;
            img(x, y, 1) = cam_bg.g;
            img(x, y, 2) = cam_bg.b;

            // Image Plane Coordinates
            float ipX = (x - (img_W / 2.0)) / img_W; /* [-0.5, 0.5]*/
            float ipY = (y - (img_H / 2.0)) / img_H; /* [-0.5, 0.5]*/

            // Hit test
            tm_rSt = TIME_NOW;
            glm::vec3 cR = cam.ray(ipX, ipY);
            fqrt::tasks::hitTestResult hrt = {
                .valid = false
            };
            int sph = -1;
            for (int csph = 0; csph < sphereCount; csph++) { // loop each sphere
                fqrt::tasks::hitTestResult chrt;
                fqrt::tasks::traceIntersectSphere(cam.position(), cR, spheres[csph], &chrt);
                if (chrt.valid) {
                    if (hrt.valid && hrt.t > chrt.t) { // found closer sphere
                        sph = csph;
                        hrt = chrt;
                    } else if (!hrt.valid) { // haven't found any sphere
                        sph = csph;
                        hrt = chrt;
                    }
                }
            }
            tm_rEn = TIME_NOW;
            tm_hitTests[p] = TIME_DURATION(tm_rSt, tm_rEn);

            if (hrt.valid) { // we got the closest sphere
                tm_rSt = TIME_NOW;
                // hit a sphere, lets use it for the color calculation
                glm::vec3 pxCol(0);
                for (int lg = 0; lg < lightCount; lg++) { // loop each light
                    bool lightObstruct = false;
                    for (int b = 0; b < sphereCount; b++) {
                        fqrt::tasks::hitTestResult hrt_l;
                        fqrt::tasks::traceIntersectSphere(hrt.pos, 
                            fqrt::tasks::buildDirRay(hrt.pos, lights[lg].pos),
                            spheres[b].pos, spheres[b].r, &hrt_l);
                        if (hrt_l.valid) { // we are obstructed if this hits
                            lightObstruct = true;
                            break;
                        }
                    }
                    if (!lightObstruct) { // if not obstructed, incorporate this light
                        float intensity = fqrt::tasks::calcLightingAtPos(lights[lg].pos, 
                            hrt.pos, hrt.nor);
                        pxCol = pxCol + intensity * lights[lg].color * spheres[sph].color;
                    }
                }
                img(x, y, 0) = (pxCol.r) * 255;
                img(x, y, 1) = (pxCol.g) * 255;
                img(x, y, 2) = (pxCol.b) * 255;
                tm_rEn = TIME_NOW;
                tm_lightTests[p] = TIME_DURATION(tm_rSt, tm_rEn);
            }
            // printf("P: %8d / %.0f @ (%04d, %04d) | H: %2.4f | L: %2.4f\r", p, img_W*img_H, x, y, tm_hitTests[p], tm_lightTests[p]);
            p++;
        }
    }
    high_resolution_clock::time_point tm_renderEnd = TIME_NOW;
    // save image
    img.save("out.bmp");
    std::cout << "\nRendered to out.bmp\n";
    std::cout << "\n------- Time Stats [s] -------" << std::endl \
        << "           load scene: " << TIME_DURATION(tm_start, tm_loaded) << std::endl \
        << "         render scene: " << TIME_DURATION(tm_renderBegin, tm_renderEnd) << std::endl \
        << "      avg depth check: " << fqrt::math::average(tm_hitTests, img_W*img_H) << std::endl \
        << "  longest depth check: " << fqrt::math::max(tm_hitTests, img_W*img_H) << std::endl \
        << "    avg light process: " << fqrt::math::average(tm_lightTests, img_W*img_H) << std::endl \
        << "longest light process: " << fqrt::math::max(tm_lightTests, img_W*img_H) << std::endl \
        << "   total program exec: " << TIME_DURATION(tm_start, tm_renderEnd) << std::endl;

    return 0;
}