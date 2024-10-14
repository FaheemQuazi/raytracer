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
#include "fqrt/tasks.hpp"
#include "fqrt/util.hpp"

using namespace std::chrono;
#define TIME_NOW high_resolution_clock::now()
#define TIME_DURATION(s, f) duration_cast<duration<double>>((f) - (s)).count()

 // load scene
#ifndef SCENE_FILE
#define SCENE_FILE argv[1]
#endif

// specify threadcount
#ifndef THREAD_COUNT
#define THREAD_COUNT atoi(argv[2])
#endif

void renderPixel(int p, int x, int y, fqrt::scene::sceneData_t *sd, fqrt::time::frameTimes_t *tm) {
    // set background color
    sd->img(x, y, 0) = sd->cam_bg.r;
    sd->img(x, y, 1) = sd->cam_bg.g;
    sd->img(x, y, 2) = sd->cam_bg.b;

    // Image Plane Coordinates
    float ipX = (x - (sd->dW / 2.0)) / sd->dW; /* [-0.5, 0.5]*/
    float ipY = (y - (sd->dH / 2.0)) / sd->dH; /* [-0.5, 0.5]*/

    // Hit test
    high_resolution_clock::time_point tm_rSt = TIME_NOW;
    glm::vec3 cR = sd->cam.ray(ipX, ipY);
    fqrt::tasks::hitTestResult hrt = {
        .valid = false
    };
    glm::vec3 cHitObjCol(0);

    for (int cmsh = 0; cmsh < sd->meshCount; cmsh++) { // loop each mesh
        for (int mT = 0; mT < sd->meshes[cmsh].count(); mT++) {
            fqrt::tasks::hitTestResult chrt;
            fqrt::tasks::traceIntersectTriangle(sd->cam.position(), cR, sd->meshes[cmsh][mT], &chrt);
            if (chrt.valid) {
                if (hrt.valid && hrt.t > chrt.t) { // found closer triangle
                    cHitObjCol = sd->meshes[cmsh][mT].n;
                    hrt = chrt;
                } else if (!hrt.valid) { // havent found anything yet
                    cHitObjCol = sd->meshes[cmsh][mT].n;
                    hrt = chrt;
                }
            }
        }
    }

    for (int csph = 0; csph < sd->sphereCount; csph++) { // loop each sphere
        fqrt::tasks::hitTestResult chrt;
        fqrt::tasks::traceIntersectSphere(sd->cam.position(), cR, sd->spheres[csph], &chrt);
        if (chrt.valid) {
            if (hrt.valid && hrt.t > chrt.t) { // found closer sphere
                cHitObjCol = sd->spheres[csph].color;
                hrt = chrt;
            } else if (!hrt.valid) { // havent found anything yet
                cHitObjCol = sd->spheres[csph].color;
                hrt = chrt;
            }
        }
    }

    for (int cpl = 0; cpl < sd->planeCount; cpl++) { // loop each plane
        fqrt::tasks::hitTestResult chrt;
        fqrt::tasks::traceIntersectPlane(sd->cam.position(), cR, sd->planes[cpl], &chrt);
        if (chrt.valid) {
            if (hrt.valid && hrt.t > chrt.t) { // found closer plane
                cHitObjCol = sd->planes[cpl].color;
                hrt = chrt;
            } else if (!hrt.valid) { // haven't found anything yet
                cHitObjCol = sd->planes[cpl].color;
                hrt = chrt;
            }
        }
    }
    high_resolution_clock::time_point tm_rEn = TIME_NOW;
    tm->hit = TIME_DURATION(tm_rSt, tm_rEn);

    if (hrt.valid) { // we got an object here
        tm_rSt = TIME_NOW;
        glm::vec3 pxCol(0);
        for (int lg = 0; lg < sd->lightCount; lg++) { // loop each light
            bool lightObstruct = false;
            for (int b = 0; b < sd->sphereCount; b++) {
                fqrt::tasks::hitTestResult hrt_l;
                fqrt::tasks::traceIntersectSphere(hrt.pos + SURF_OFFSET_SPHERE*hrt.nor, // move up a hair off the surface
                    fqrt::tasks::buildDirRay(hrt.pos + SURF_OFFSET_SPHERE*hrt.nor, sd->lights[lg].pos),
                    sd->spheres[b], &hrt_l);
                if (hrt_l.valid) { // we are obstructed if this hits
                    lightObstruct = true;
                    break;
                }
            }
            if (!lightObstruct && sd->meshCount > 0) { // if not obstructed check meshes
                for (int b = 0; b < sd->meshCount; b++) {
                    fqrt::tasks::hitTestResult hrt_l;
                    for (int mt = 0; mt < sd->meshes[b].count(); mt++) {
                        fqrt::tasks::traceIntersectTriangle(hrt.pos + SURF_OFFSET_PLANE*hrt.nor, // move up a hair off the surface
                            fqrt::tasks::buildDirRay(hrt.pos + SURF_OFFSET_PLANE*hrt.nor, sd->lights[lg].pos),
                            sd->meshes[b][mt], &hrt_l);
                        if (hrt_l.valid) { // we are obstructed if this hits
                            lightObstruct = true;
                            break;
                        }
                    }
                    if (hrt_l.valid) { // break out if any triangle in this mesh hits
                        lightObstruct = true;
                        break;
                    }
                }
            }
            if (!lightObstruct && sd->planeCount > 0) { // if not obstructed check planes
                for (int b = 0; b < sd->planeCount; b++) {
                    fqrt::tasks::hitTestResult hrt_l;
                    fqrt::tasks::traceIntersectPlane(hrt.pos + SURF_OFFSET_PLANE*hrt.nor, // move up a hair off the surface
                        fqrt::tasks::buildDirRay(hrt.pos + SURF_OFFSET_PLANE*hrt.nor, sd->lights[lg].pos),
                        sd->planes[b], &hrt_l);
                    if (hrt_l.valid) { // we are obstructed if this hits
                        lightObstruct = true;
                        break;
                    }
                }
            }
            if (!lightObstruct) { // if not obstructed, incorporate this light
                float intensity = fqrt::tasks::calcLightingAtPos(sd->lights[lg].pos, 
                    hrt.pos, hrt.nor);
                pxCol = pxCol + intensity * sd->lights[lg].color * cHitObjCol;
            }
        }
        sd->img(x, y, 0) = (pxCol.r) * 255;
        sd->img(x, y, 1) = (pxCol.g) * 255;
        sd->img(x, y, 2) = (pxCol.b) * 255;
        tm_rEn = TIME_NOW;
        tm->light = TIME_DURATION(tm_rSt, tm_rEn);
    }
}

int main(int argc, char* argv[])
{
    std::cout << "Hello FQRT\n";
    if (argc < 3) {
        std::cerr << "Specify scene file and number of threads!\n";
        return 1;
    }
    std::cout << "Scene File: " << SCENE_FILE << "\n";
    std::cout << "Thread Limit: " << THREAD_COUNT << "\n";

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

    // meshes
    sd.meshCount = scene.count("mesh");
    if (sd.meshCount > 0) {
        // split the scene path string to get directory
        std::vector<std::string> path = fqrt::files::SplitPath(SCENE_FILE);
        path.pop_back(); // remove the file name
        sd.meshes = new tira::simplemesh[sd.meshCount];
        for (int i = 0; i < sd.meshCount; i++) {
            path.push_back(scene.get<std::string>("mesh", i, 0));
            std::string meshPath = fqrt::files::JoinPath(path);
            sd.meshes[i].load(meshPath);
        }
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
        << " - Number of Meshes: " << sd.meshCount << std::endl \
        << " - Number of Lights: " << sd.lightCount << std::endl;

    // Time tracking stuff    
    fqrt::time::frameTimes_t *tm_pixelTimes = (fqrt::time::frameTimes_t*)malloc(sizeof(fqrt::time::frameTimes_t) * sd.dH * sd.dW);

    // Begin Render Pass
    std::cout << "begin render...";
    high_resolution_clock::time_point tm_renderBegin = TIME_NOW;
    int p = 0;
    if (THREAD_COUNT > 1) {
        std::thread th[THREAD_COUNT];    
        for (int y = 0; y < sd.dH; y++) {
            for (int x = 0; x < sd.dW; x++) {
                if (p >= THREAD_COUNT) { // only join if we've passed the array size
                    th[p%THREAD_COUNT].join();
                }                
                th[p%THREAD_COUNT] = std::thread(renderPixel, p, x, y, &sd, &tm_pixelTimes[p]);
                p++;
            }
        }
        for (int t = 0; t < THREAD_COUNT; t++) {
            th[t].join();
        }
    } else {
        for (int y = 0; y < sd.dH; y++) {
            for (int x = 0; x < sd.dW; x++) {
                renderPixel(p, x, y, &sd, &tm_pixelTimes[p]);
                p++;
            }
        }
    }
    high_resolution_clock::time_point tm_renderEnd = TIME_NOW;
    // save image
    sd.img.save("out.bmp");
    // for stats
    double *ht = fqrt::time::FTgetHits(tm_pixelTimes, sd.dW*sd.dH);
    double *lt = fqrt::time::FTgetLights(tm_pixelTimes, sd.dW*sd.dH);
    std::cout << "\nRendered to out.bmp\n";
    std::cout << "\n--------- Time Stats [s] ---------" << std::endl \
        << "           load scene: " << TIME_DURATION(tm_start, tm_loaded) << std::endl \
        << "         render scene: " << TIME_DURATION(tm_renderBegin, tm_renderEnd) << std::endl \
        << "      avg depth check: " << fqrt::math::average(ht, sd.dW*sd.dH) << std::endl \
        << "  longest depth check: " << fqrt::math::max(ht, sd.dW*sd.dH) << std::endl \
        << "    avg light process: " << fqrt::math::average(lt, sd.dW*sd.dH) << std::endl \
        << "longest light process: " << fqrt::math::max(lt, sd.dW*sd.dH) << std::endl \
        << "   total program exec: " << TIME_DURATION(tm_start, tm_renderEnd) << std::endl;

    return 0;
}