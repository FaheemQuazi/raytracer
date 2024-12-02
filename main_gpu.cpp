#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include "tira/parser.h"
#include "tira/graphics/camera.h"
#include "tira/graphics/shapes/simplemesh.h"
#include "tira/image.h"
#include "fqrt/objects.hpp"
#include "fqrt/scene.hpp"
#include "fqrt/tasks.hpp"
#include "fqrt/util.hpp"
#include "rt.cuh"

using namespace std::chrono;
#define TIME_NOW high_resolution_clock::now()
#define TIME_DURATION(s, f) duration_cast<duration<double>>((f) - (s)).count()

 // load scene
#ifndef SCENE_FILE
#define SCENE_FILE argv[1]
#endif

int main(int argc, char** argv) {
    std::cout << "Hello FQRT\n";
    if (argc < 2) {
        std::cerr << "Specify scene file!\n";
        return 1;
    }
    std::cout << "Scene File: " << SCENE_FILE << "\n";

    // timing functions
    high_resolution_clock::time_point tm_start = TIME_NOW;

    tira::parser scene(SCENE_FILE);
    fqrt::scene::sceneData_t sd;
    sd.cam = tira::camera();

    // output image
    sd.dW = scene.get<float>("resolution", 0);
    sd.dH = scene.get<float>("resolution", 1);
    sd.dC = 3;
    sd.img = tira::image<unsigned char>(sd.dW, sd.dH, sd.dC);

    // configure camera
    sd.cam.position(scene.get<float>("camera_position", 0), scene.get<float>("camera_position", 1), scene.get<float>("camera_position", 2));
    sd.cam.lookat(scene.get<float>("camera_look", 0), scene.get<float>("camera_look", 1), scene.get<float>("camera_look", 2));
    sd.cam.up(glm::vec3(scene.get<float>("camera_up", 0), scene.get<float>("camera_up", 1)*-1, scene.get<float>("camera_up", 2))); // why -1?
    sd.cam.fov(scene.get<float>("camera_fov", 0));
    sd.cam_bg = glm::vec3(scene.get<float>("background", 0), scene.get<float>("background", 1), scene.get<float>("background", 2));

    // allocate spheres
    sd.sphereCount = scene.count("sphere");
    if (sd.sphereCount > 0) {
        sd.spheres = (fqrt::objects::sphere *)malloc(sizeof(fqrt::objects::sphere) * sd.sphereCount);
        for (int i = 0; i < sd.sphereCount; i++) {
            sd.spheres[i].r = scene.get<float>("sphere", i, 0);
            sd.spheres[i].pos.x = scene.get<float>("sphere", i, 1);
            sd.spheres[i].pos.y = scene.get<float>("sphere", i, 2);
            sd.spheres[i].pos.z = scene.get<float>("sphere", i, 3);
            sd.spheres[i].color.r = scene.get<float>("sphere", i, 4);
            sd.spheres[i].color.g = scene.get<float>("sphere", i, 5);
            sd.spheres[i].color.b = scene.get<float>("sphere", i, 6);
        }
    } else {
        printf("Warning: No Spheres in scene!\n");
    }
    
    // allocate planes
    sd.planeCount = scene.count("plane");
    if (sd.planeCount > 0) {
        sd.planes = (fqrt::objects::plane *)malloc(sizeof(fqrt::objects::plane) * sd.planeCount);
        for (int i = 0; i < sd.planeCount; i++) {
            sd.planes[i].pos.x = scene.get<float>("plane", i, 0);
            sd.planes[i].pos.y = scene.get<float>("plane", i, 1);
            sd.planes[i].pos.z = scene.get<float>("plane", i, 2);
            sd.planes[i].norm.x = scene.get<float>("plane", i, 3);
            sd.planes[i].norm.y = scene.get<float>("plane", i, 4);
            sd.planes[i].norm.z = scene.get<float>("plane", i, 5);
            sd.planes[i].color.r = scene.get<float>("plane", i, 6);
            sd.planes[i].color.g = scene.get<float>("plane", i, 7);
            sd.planes[i].color.b = scene.get<float>("plane", i, 8);
        }
    } else {
        printf("Warning: No planes in scene!\n");
    }

    // allocate lights
    sd.lightCount = scene.count("light");
    if (sd.lightCount > 0) {
        sd.lights = (fqrt::objects::light *)malloc(sizeof(fqrt::objects::light) * sd.lightCount);
        for (int i = 0; i < sd.lightCount; i++) {
            sd.lights[i].pos.x = scene.get<float>("light", i, 0);
            sd.lights[i].pos.y = scene.get<float>("light", i, 1);
            sd.lights[i].pos.z = scene.get<float>("light", i, 2);
            sd.lights[i].color.r = scene.get<float>("light", i, 3);
            sd.lights[i].color.g = scene.get<float>("light", i, 4);
            sd.lights[i].color.b = scene.get<float>("light", i, 5);
        }
    } else {
        printf("Error: No Lights in scene!\n");
        return 1;
    }
    
    high_resolution_clock::time_point tm_loaded = TIME_NOW;
    // print what we've loaded so far
    std::cout << "Loaded:\n" \
        << " - Camera: " << sd.cam \
        << " - Render Image: " << sd.dW << " X " << sd.dH << std::endl\
        << " - Number of Spheres: " << sd.sphereCount << std::endl \
        << " - Number of Planes: " << sd.planeCount << std::endl \
        << " - Number of Lights: " << sd.lightCount << std::endl;

    if (!cuda_avail()) {
        std::cout << "ERROR: NO CUDA GPU FOUND!\n";
        return 1;
    }
    
    std::cout << "Start Render...\n";
    cuda_hello();

    return 0;
}