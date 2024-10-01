Raytrace Assignment - ECE 6397
Faheem Quazi

-------------
How to build:
-------------
I primarily followed the "hello vcpkg" tutorial for building.
This repository is also configured in VSCode to auto-configure
if you have the CMake Tools extension installed. I'm including
the below instructions just in case that's not desired:

0) Follow step 1 and 2 of the tutorial
    https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-bash
    you basically need to set up VCPKG_ROOT
1) cd into this directory and clone submodules to download TIRA:
    `git submodule update --init --recursive` 
2) Configure the CMake Build
    `cmake --preset=release`
3) Build the app
    `cmake --build build/release`
4) the executable will be in `build/release`

-----------
How to run:
-----------
The executable requires the path to a `.scene` file. You can
see examples of them in the `assets/` directory.

Example:
    `raytracer ./assets/basic.scene`

The program will print the output times and render the image
in `out.bmp`.

