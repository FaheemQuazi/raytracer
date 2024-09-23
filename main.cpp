#include <iostream>
#include "tira/parser.h"
#include "tira/graphics/camera.h"
#include "tira/image.h"
#include "fqrt/objects.hpp"
#include "fqrt/tasks.hpp"

 // load scene
#define SCENE_FILE "assets/basic.scene"
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

int main()
{
    std::cout << "Hello FQRT\n";

    // configure camera
    cam.position(scene.get<float>("camera_position", 0), scene.get<float>("camera_position", 1), scene.get<float>("camera_position", 2));
    cam.lookat(scene.get<float>("camera_look", 0), scene.get<float>("camera_look", 1), scene.get<float>("camera_look", 2));
    cam.up(glm::vec3(scene.get<float>("camera_up", 0), scene.get<float>("camera_up", 1), scene.get<float>("camera_up", 2)));
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
    
    // print what we've loaded so far
    std::cout << "Loaded:\n" \
        /*Scene Name*/          << " - Scene: " << SCENE_FILE << std::endl \
        /*Camera*/              << " - Camera: " << cam \
        /*Render Dimensions*/   << " - Render Image: " << img_W << " X " << img_H << std::endl\
        /*Sphere Count*/        << " - Number of Spheres: " << sphereCount << std::endl \
        /*Light Count*/         << " - Number of Lights:" << lightCount << std::endl;

    // Begin Render Pass
    for (int y = 0; y < img_H; y++) {
        for (int x = 0; x < img_W; x++) {
            // Image Plane Coordinates
            float ipX = (x - (img_W / 2.0)) / img_W; /* [-0.5, 0.5]*/
            float ipY = (y - (img_H / 2.0)) / img_H; /* [-0.5, 0.5]*/
            glm::vec3 cR = cam.ray(ipX, ipY);
            for (int sph = 0; sph < sphereCount; sph++) {
                if (fqrt::tasks::testIntersectSphere(cam.position(), cR, spheres[sph].pos, spheres[sph].r)) {
                    img(x, y, 0) = 255;
                    img(x, y, 1) = 255;
                    img(x, y, 2) = 255;
                    break;
                } else {
                    img(x, y, 0) = 0;
                    img(x, y, 1) = 0;
                    img(x, y, 2) = 0;
                }
            }
        }
    }
    // save image
    img.save("out.bmp");

    return 0;
}