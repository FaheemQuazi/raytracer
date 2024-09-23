#include <iostream>
#include "tira/parser.h"
#include "tira/graphics/camera.h"
#include "tira/image.h"
#include "fqrt/objects.hpp"

 // load scene
#define SCENE_FILE "assets/basic.scene"
tira::parser scene(SCENE_FILE);

// scene objects
tira::camera cam;
fqrt::objects::sphere *spheres;
fqrt::objects::light *lights;

// output image
int img_W = scene.get<int>("resolution", 0);
int img_H = scene.get<int>("resolution", 1);
int img_C = 3;
tira::image<unsigned char> img(img_W, img_H, img_C);

int main()
{
    std::cout << "Hello FQRT\n";

    // configure camera
    cam.position(scene.get<float>("camera_position", 0), scene.get<float>("camera_position", 1), scene.get<float>("camera_position", 2));
    cam.lookat(scene.get<float>("camera_look", 0), scene.get<float>("camera_look", 1), scene.get<float>("camera_look", 2));
    cam.up(glm::vec3(scene.get<float>("camera_up", 0), scene.get<float>("camera_up", 1), scene.get<float>("camera_up", 2)));
    cam.fov(scene.get<float>("camera_fov", 0));

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
    
    // print what we've loaded so far
    std::cout << "Loaded:\n" \
        /*Scene Name*/          << " - Scene: " << SCENE_FILE << "\n" \
        /*Render Dimensions*/   << " - Render Image: " << img_W << " X " << img_H << "\n"\
        /*Sphere Count*/        << " - Number of Spheres: " << sphereCount << "\n" \
        /*Light Count*/         << " - Number of Lights:" << lightCount << "\n";

    return 0;
}