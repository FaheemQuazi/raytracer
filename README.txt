Raytrace Assignment - ECE 6397
Faheem Quazi

-------------
How to build:
-------------
HIGHLY RECOMMEND using the Dev Container configuration to 
automate these setup steps.

I primarily followed the "hello vcpkg" tutorial for building.
This repository is also configured in VSCode to auto-configure
if you have the CMake Tools extension installed. 

0) Ensure you have NVIDIA CUDA Toolkit installed.
    Follow step 1 and 2 of the tutorial at:
    https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-bash
    you basically set up VCPKG_ROOT so vcpkg works.
1) cd into this directory and clone submodules to download TIRA:
    `git submodule update --init --recursive` 
2) Configure the CMake Build
    `cmake --preset=release`
3) Build the app
    `cmake --build build/release`
4) the executables will be in `build/release`

-----------
How to run:
-----------
The executables requires the path to a `.scene` file and a core count. 
You can see examples of scenes in the `assets/` directory.

Example (CPU):
    `raytracer ./assets/basic.scene 4`
The program will print the output times and render the image
in `out.bmp` using 4 cores

Example (GPU):
    `cudatracer ./assets/basic.scene`
The program will print the output times and render the image
in `out.bmp` using your GPU. NOTE that you will need a
NVIDIA GPU installed in your system to run this.

