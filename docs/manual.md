# MegaMol Manual

<!-- TOC -->

## Contents

<<<<<<< HEAD
<<<<<<< HEAD
- [Overview](#overview)
=======
- [Overview](#overview)
<<<<<<< HEAD
    - [License](#license)
>>>>>>> d6e034ef9 (docu toc fix)
=======
>>>>>>> 55301faa9 (docu ...)
- [Installation and Setup](#installation-and-setup)
    - [Building from Source](#building-from-source)
        - [Microsoft Windows](#microsoft-windows)
        - [Linux (Ubuntu)](#linux-ubuntu)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    - [Command Line Arguments](#command-line-arguments)        
    - [Configuration File](#configuration-file)
=======
    - [Configuration](#configuration)
>>>>>>> d6e034ef9 (docu toc fix)
=======
    - [Commandline Arguments](#commandline-arguments)        
=======
    - [Command Line Arguments](#command-line-arguments)        
>>>>>>> ee6adca4d (docu)
    - [Configuration File](#configuration-file)
>>>>>>> 94da8d87e (docu)
        - [General Settings](#general-settings)
        - [Logging](#logging)
        - [Application, Shaders and Resources](#application-shaders-and-resources) 
        - [Plugins](#plugins) 
        - [Global Settings](#global-settings) 
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    - [Test Installation](#test-installation) 
        - [Examples](#examples) 
=======
        - [Tests](#tests) 
=======
        - [Test Installation](#test-installation) 
>>>>>>> 9f08ae8ac (updted cineamtic and ospray docu)
- [Load and Create Projects](#load-and-create-projects)
>>>>>>> d6e034ef9 (docu toc fix)
=======
    - [Test Installation](#test-installation) 
        - [Examples](#examples) 
<<<<<<< HEAD
    - [Project Files](#project-files) 
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
=======
>>>>>>> e7bb59b8f (docu)
- [Viewing Data Sets](#viewing-data-sets)
    - [Modules, Views and Calls](#modules-views-and-calls) 
        - [Modules and Calls](#modules-and-calls) 
        - [Views](#views) 
<<<<<<< HEAD
<<<<<<< HEAD
    - [View Interaction](#view-interaction) 
- [Project Files](#project-files)     
- [Making High-Resolution Screenshots](#making-high-resolution-screenshots) 
- [Making Simple Videos](#making-simple-videos) 
- [Reproducibility](#reproducibility) 
<!-- 
- [Jobs](#jobs)
    - [Job Instance](#job-instance) 
    - [Converting to MMPLD](#converting-to-mmpld) 
-->

<!-- /TOC -->

=======
- [Contents](#contents)
    - [Overview](#overview)
        - [License](#license)
    - [Installation and Setup](#installation-and-setup)
        - [Building from Source](#building-from-source)
            - [Microsoft Windows](#microsoft-windows)
            - [Linux (Ubuntu)](#linux-ubuntu)
        - [Configuration](#configuration)
            - [General Settings](#general-settings)
            - [Logging](#logging)
            - [Application, Shaders and Resources](#application-shaders-and-resources) 
            - [Plugins](#plugins) 
            - [Global Settings](#global-settings) 
            - [Tests](#tests) 
    - [Load and Create Projects](#load-and-create-projects)
    - [Viewing Data Sets](#viewing-data-sets)
        - [Modules, Views and Calls](#modules-views-and-calls) 
            - [Modules and Calls](#modules-and-calls) 
            - [Views](#views) 
        - [Project Files](#project-files) 
        - [View Interaction](#view-interaction) 
        - [Making High-Resolution Screenshots](#making-high-resolution-screenshots) 
        - [Reproducibility](#reproducibility) 
        - [Making Simple Videos](#making-simple-videos) 
    - [Jobs: Converting Data](#jobs)
        - [Job Instance](#job-instance) 
        - [Converting to MMPLD](#converting-to-mmpld) 
    <!-- - [Advanced Usage](#advanced-usage) -->
=======
    - [Project Files](#project-files) 
=======
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
    - [View Interaction](#view-interaction) 
- [Project Files](#project-files)     
- [Making High-Resolution Screenshots](#making-high-resolution-screenshots) 
- [Making Simple Videos](#making-simple-videos) 
- [Reproducibility](#reproducibility) 
<!-- 
- [Jobs](#jobs)
    - [Job Instance](#job-instance) 
    - [Converting to MMPLD](#converting-to-mmpld) 
<<<<<<< HEAD
<<<<<<< HEAD
<!-- - [Advanced Usage](#advanced-usage) -->
>>>>>>> d6e034ef9 (docu toc fix)
=======
=======
-->
<<<<<<< HEAD
>>>>>>> ee6adca4d (docu)
- [Reproducibility](#reproducibility) 
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
=======
>>>>>>> e7bb59b8f (docu)

<!-- /TOC -->


<<<<<<< HEAD
<<<<<<< HEAD
<!-- ###################################################################### -->
<a name="overview"></a>
>>>>>>> 800f17d5c (manual update)

<!-- ###################################################################### -->
-----
<<<<<<< HEAD
=======
[//]: # (######################################################################)
>>>>>>> 4fa438626 (manual update ...)
=======
<!-- ###################################################################### -->
>>>>>>> 0ae2f4429 (manual update ...)
=======
>>>>>>> 6668c26ff (docu toc)
## Overview

MegaMol is a visualization middleware used to visualize point-based molecular datasets.
The MegaMol project was started in the Collaborative Research Center 716, subproject D.3, at the Visualization Research Center (VISUS), University of Stuttgart, Germany.
Today, it is governed by a number of teams at the TU Dresden and the University of Stuttgart.

The goal of the project is to provide a software base for visualization research and to provide a stable environment to deploy newest visualization prototypes to application domain researchers. MegaMol is not a visualization tool. MegaMol is a platform for visualization research.
<<<<<<< HEAD
<<<<<<< HEAD
Visit the project [website](https://megamol.org/ "MegaMol Homepage") for downloads and more information.

<<<<<<< HEAD
**If you faced any trouble during installation or if you have any further questions concerning MegaMol, we encourage you to contact the developer team by opening an [issue](https://github.com/UniStuttgart-VISUS/megamol/issues/new) on github!**

<<<<<<< HEAD
=======
<!-- ---------------------------------------------------------------------- -->
=======
[//]: # (----------------------------------------------------------------------) 
>>>>>>> 4fa438626 (manual update ...)
=======
Visit the project [website](https://github.com/UniStuttgart-VISUS/megamol.git "MegaMol Homepage") for downloads and more information.

<!-- ---------------------------------------------------------------------- -->
>>>>>>> 0ae2f4429 (manual update ...)
### License

MegaMol is freely and publicly available as open source following the terms of the BSD License.
Copyright (c) 2015, MegaMol Team TU Dresden, Germany Visualization Research Center, University of Stuttgart (VISUS), Germany
Alle Rechte vorbehalten.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the MegaMol Team, TU Dresden, University of Stuttgart, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE MEGAMOL TEAM "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE MEGAMOL TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=======
Visit the project [website](https://megamol.org/ "MegaMol Homepage") for downloads and more information.
>>>>>>> 55301faa9 (docu ...)


<<<<<<< HEAD

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<!-- ###################################################################### -->
<a name="installation-and-setup"></a>
>>>>>>> 800f17d5c (manual update)
=======
**If you faced any trouble during installation or if you have any furhter questions concerning MegaMol, we encourage you to contact the developer team by opening an [issue](https://github.com/UniStuttgart-VISUS/megamol/issues/new) on github!**

>>>>>>> bdc293c78 (docu)

=======
>>>>>>> 8e58073aa (...)
<!-- ###################################################################### -->
-----
<<<<<<< HEAD
=======
[//]: # (######################################################################) 
>>>>>>> 4fa438626 (manual update ...)
=======
<!-- ###################################################################### -->
>>>>>>> 0ae2f4429 (manual update ...)
=======
>>>>>>> 6668c26ff (docu toc)
## Installation and Setup

<<<<<<< HEAD
<<<<<<< HEAD
This chapter discusses installation and setup of MegaMol from source code.
MegaMol targets Microsoft Windows (Windows 7 or newer, x64) and Linux (x64) as supported environments.
=======
This chapter discusses installation and setup of MegaMol, either from the pre-built binary packages or the source code.
The latter is, however, meant for experienced users. MegaMol targets Microsoft Windows (Windows 7 or newer, x86 and x64) and Linux (x64) as supported environments.
>>>>>>> 800f17d5c (manual update)
=======
This chapter discusses installation and setup of MegaMol from source code.
MegaMol targets Microsoft Windows (Windows 7 or newer, x64) and Linux (x64) as supported environments.
>>>>>>> 88e3119f9 (docu)
Currently, Ubuntu is used as Linux distribution for development.
Further platforms are not considered during the development. 
While MegaMol might work on further platforms, the development team will currently not grant any support for problems with these environments.

<!-- ---------------------------------------------------------------------- -->
### Building from Source

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
It is recommended to use the latest [release 1.3](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3) version of the source code.
All bleeding edge features are available in the current [main branch](https://github.com/UniStuttgart-VISUS/megamol/tree/master).
<<<<<<< HEAD
=======
Download a local copy of the latest [release](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3). 
=======
The latest release [release](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3). 
>>>>>>> 88e3119f9 (docu)
(Using the current main [branch](https://github.com/UniStuttgart-VISUS/megamol.git) is not recommended, since there is a lot of more or less untested bleeding edge stuff going on.)
=======
It is recommende to use the latest [release](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3) version of the source code. 
>>>>>>> 8b2342edf (docu)

**Note**: 
The *OSPRay plugin* is currently disabled by default. 
See the plugins' [readme](https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/OSPRay_plugin/Readme.md) for additional instructions on how to enable it.

<!-- ---------------------------------------------------------------------- -->
#### Microsoft Windows
>>>>>>> 800f17d5c (manual update)
=======
It is recommended to use the latest [release](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3) version of the source code.
=======
It is recommended to use the latest [release 1.3](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3) version of the source code.
>>>>>>> a1cbe09fa (docu ...)
All bleeding edge features are available in the [main branch](https://github.com/UniStuttgart-VISUS/megamol/tree/master).
>>>>>>> 55301faa9 (docu ...)
=======
>>>>>>> ee6adca4d (docu)

**Note**: 
The *OSPRay plugin* is currently disabled by default. 
See the plugins' [readme](https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/OSPRay_plugin/Readme.md) for additional instructions on how to enable it.

<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
#### Microsoft Windows

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 94da8d87e (docu)
1. Download and unzip the source code of the latest [release](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3) into a new directory (e.g. `megamol`).
2. You have to install [CMake](https://cmake.org/), and load the `CMakeLists.txt` present in the root directory of the repository.
3. Create a new `build` directory.
4. As generator, it is recommended to use the latest version of [Visual Studio](https://visualstudio.microsoft.com/downloads/) (Community Edition is free to use) with default native compilers and for the platform x64.
5. Next, click `Configure` a few times (until all red entries disappear).
6. Change the `CMAKE_INSTALL_PREFIX` in order to change the destination directory of the installed files and configure once more.
7. Then click `Generate` to generate the build files.
8. The configuration creates a `megamol.sln` file inside the build directory.
9. Open the `megamol.sln` file with *Visual Studio*. 
10. Use the `ALL_BUILD` target to build MegaMol.
11. Afterwards, use the `INSTALL` target to create your MegaMol installation.
12. The binary `megamol.exe` is located in the default installation path `../megamol/build/install/bin`.
<<<<<<< HEAD
=======
- For Windows, you have to install [CMake](https://cmake.org/), and load the `CMakeLists.txt` present in the root directory of the repository. 
- Next, click `Configure` a few times (until all red entries disappear).
- Then click `Generate` to generate the build files.
- It is recommended to use **Visual Studio 16 2019 (platform x64)** with default native compilers as generator.
>>>>>>> d85984bb9 (docu)
=======
- Download and unzip the source code of the latest [release](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3) into a new folder (e.g. `megamol`).
=======
- Download and unzip the source code of the latest [release](https://github.com/UniStuttgart-VISUS/megamol/releases/tag/v1.3) into a new directory (e.g. `megamol`).
>>>>>>> 8b2342edf (docu)
- You have to install [CMake](https://cmake.org/), and load the `CMakeLists.txt` present in the root directory of the repository.
- Create a new `build` directory.
- AS generator, it is recommended to use the latest version of [Visual Studio](https://visualstudio.microsoft.com/downloads/) (Community Edition is free to use) with default native compilers and for the platform x64.
- Next, click `Configure` a few times (until all red entries disappear).
- Change the `CMAKE_INSTALL_PREFIX` in order to change the destination directory of the installed files.
- Then click `Generate` to generate the build files.
- The configuration creates a `megamol.sln` file inside the build directory.
- Open the `sln` file with *Visual Studio*. 
- Use the `ALL_BUILD` target to build MegaMol.
- Afterwards, use the `INSTALL` target to create your MegaMol installation.
<<<<<<< HEAD
- The binary `megamol.exe` is located in the default installation path `../megamol/build/install/bin`
>>>>>>> 88e3119f9 (docu)
=======
- The binary `megamol.exe` is located in the default installation path `../megamol/build/install/bin`.
>>>>>>> 06b190844 (docu)
=======
>>>>>>> 94da8d87e (docu)

![CMake Windows](pics/cmake_windows.png)
*Screenshot of `cmake-gui` after generating build files.*

<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
#### Linux (Ubuntu)

Since the full support of some C++17 functionality is required (e.g. *std::filesystem*), a `gcc` version equal or greater than **8** is required (with `CMAKE_CXX_FLAGS` appended by `--std=c++17`). 
**Latest tested version:**
=======
<center>
<a name="cmake_windows"></a>
<img src="pics/cmake_windows.png" alt="CMake Windows" style="width: 768px;"/>
</center>
=======
![CMake Windows](pics/cmake_windows.png)
<<<<<<< HEAD
Screenshot of `cmake-gui` after generating build files.
>>>>>>> f5eec258f (manual update ....)
=======
*Screenshot of `cmake-gui` after generating build files.*
>>>>>>> f42be6ab4 (...)

<!-- ---------------------------------------------------------------------- -->
#### Linux (Ubuntu)

<<<<<<< HEAD
<<<<<<< HEAD
Since the full support of some C++17 functionality is required (e.g. std::filesystem) a `gcc` version equal or greater than **8** is required (with `CMAKE_CXX_FLAGS` appended by `--std=c++17`).

<<<<<<< HEAD
Latest Test:
>>>>>>> 800f17d5c (manual update)
=======
=======
Since the full support of some C++17 functionality is required (e.g. std::filesystem), a `gcc` version equal or greater than **8** is required (with `CMAKE_CXX_FLAGS` appended by `--std=c++17`).
<<<<<<< HEAD
>>>>>>> d85984bb9 (docu)
=======

>>>>>>> 8b2342edf (docu)
Latest tested version:
>>>>>>> f5eec258f (manual update ....)
=======
Since the full support of some C++17 functionality is required (e.g. *std::filesystem*), a `gcc` version equal or greater than **8** is required (with `CMAKE_CXX_FLAGS` appended by `--std=c++17`). 
**Latest tested version:**
>>>>>>> a1cbe09fa (docu ...)

    $ cat /proc/version
    Linux version 5.8.0-41-generic (buildd@lgw01-amd64-003) (gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0, GNU ld (GNU Binutils for Ubuntu) 2.34) #46~20.04.1-Ubuntu SMP Mon Jan 18 17:52:23 UTC 2021

<<<<<<< HEAD
<<<<<<< HEAD
1. As prerequisites, following packages from the repository are required:
=======
- As prerequisites, following packages from the repository are required:
>>>>>>> 88e3119f9 (docu)

<<<<<<< HEAD
<<<<<<< HEAD
    `$ sudo apt install cmake-curses-gui git libgl1-mesa-dev libncurses5-dev uuid-dev libexpat-dev libunwind-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libglu1-mesa-dev`

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
2. First, download the source code from GitHub:
=======
```
    $ sudo apt install cmake-curses-gui git libgl1-mesa-dev libncurses5-dev uuid-dev libexpat-dev libunwind-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libglu1-mesa-dev
```
>>>>>>> aa0faee09 (docu)

<<<<<<< HEAD
<<<<<<< HEAD
    `$ mkdir megamol`
    `$ git clone https://github.com/UniStuttgart-VISUS/megamol.git megamol/`
    `$ cd megamol/`
=======
    $ sudo apt install cmake-curses-gui git libgl1-mesa-dev libncurses5-dev uuid-dev libexpat-dev libunwind-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libglu1-mesa-dev
>>>>>>> 8b2342edf (docu)

<<<<<<< HEAD
3. Checkout the latest release:
=======
    `$ sudo apt install cmake-curses-gui git libgl1-mesa-dev libncurses5-dev uuid-dev libexpat-dev libunwind-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libglu1-mesa-dev`
>>>>>>> 8e58073aa (...)
=======
    $ sudo apt install cmake-curses-gui git libgl1-mesa-dev libncurses5-dev uuid-dev libexpat-dev libunwind-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libglu1-mesa-dev
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
=======
1. As prerequisites, following packages from the repository are required:
>>>>>>> 94da8d87e (docu)

    `$ git checkout tags/v1.3 -b latest_release`
=======
First, download the software package from GitHub:
=======
- First, download the software package from GitHub:
>>>>>>> 88e3119f9 (docu)

<<<<<<< HEAD
<<<<<<< HEAD
    $ mkdir megamol; git clone https://github.com/UniStuttgart-VISUS/megamol.git megamol
>>>>>>> 800f17d5c (manual update)
=======
    $ mkdir megamol; git clone https://github.com/UniStuttgart-VISUS/megamol.git megamol/
>>>>>>> 0ae2f4429 (manual update ...)

<<<<<<< HEAD
<<<<<<< HEAD
4. Create a build directory and switch to it:
=======
- Create a build folder and switch to it:
>>>>>>> 88e3119f9 (docu)
=======
=======
    `$ mkdir megamol; git clone https://github.com/UniStuttgart-VISUS/megamol.git megamol/`
=======
=======
    `$ sudo apt install cmake-curses-gui git libgl1-mesa-dev libncurses5-dev uuid-dev libexpat-dev libunwind-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libglu1-mesa-dev`

<<<<<<< HEAD
>>>>>>> a3a2b3f57 (docu)
- First, download the source code from GitHub:
=======
2. First, download the source code from GitHub:
>>>>>>> 94da8d87e (docu)

    `$ mkdir megamol`
    `$ git clone https://github.com/UniStuttgart-VISUS/megamol.git megamol/`
    `$ cd megamol/`

3. Checkout the lastest release:

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    `$ git checkout tags/v1.3 -b latest_release`
>>>>>>> a1cbe09fa (docu ...)
=======
    $ git checkout tags/v1.3 -b latest_release
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
=======
```
    $ git checkout tags/v1.3 -b latest_release
```
>>>>>>> aa0faee09 (docu)
=======
    `$ git checkout tags/v1.3 -b latest_release`
>>>>>>> a3a2b3f57 (docu)

<<<<<<< HEAD
>>>>>>> 8e58073aa (...)
- Create a build directory and switch to it:
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 8b2342edf (docu)
=======
4. Create a build directory and switch to it:
>>>>>>> 94da8d87e (docu)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    `$ mkdir build; cd build`
=======
    $ cd megamol; mkdir build; cd build
>>>>>>> 800f17d5c (manual update)
=======
    `$ cd megamol; mkdir build; cd build`
>>>>>>> 8e58073aa (...)

<<<<<<< HEAD
5. Check for required dependencies:
    
    `$ cmake ..`
=======
=======
    $ mkdir build; cd build
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
=======

<<<<<<< HEAD
<<<<<<< HEAD
```
    $ mkdir build; cd build
```
>>>>>>> aa0faee09 (docu)
=======
    `$ mkdir build; cd build`
>>>>>>> a3a2b3f57 (docu)

    `$ mkdir build; cd build`
>>>>>>> a1cbe09fa (docu ...)

<<<<<<< HEAD
6. Start the ncurses gui for cmake:
=======
- Check for required dependencies:
    
    `$ cmake ..`

- Start the ncurses gui for cmake:
>>>>>>> 88e3119f9 (docu)
=======
5. Check for required dependencies:
    
    `$ cmake ..`

6. Start the ncurses gui for cmake:
>>>>>>> 94da8d87e (docu)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    `$ ccmake .`
<<<<<<< HEAD
=======
    $ ccmake .
=======
>>>>>>> 8e58073aa (...)
=======
    $ ccmake .
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
=======
```
    $ ccmake .
```
>>>>>>> aa0faee09 (docu)
=======
    `$ ccmake .`
>>>>>>> a3a2b3f57 (docu)

    - Configure the project repeatedly using `c` (and `e`) until no more changes are marked. 
    - Change the `CMAKE_INSTALL_PREFIX` in order to change the destination directory of the installed files.
    - Then hit `g` to generate the build files.

7. On the console prompt, start the building:

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    $ make && make install
>>>>>>> ba7df3ade (manual update)

    - Configure the project repeatedly using `c` (and `e`) until no more changes are marked. 
    - Change the `CMAKE_INSTALL_PREFIX` in order to change the destination directory of the installed files.
    - Then hit `g` to generate the build files.

<<<<<<< HEAD
7. On the console prompt, start the building:

    `$ make && make install`

    - Hint: Use the `-j` option for `make` to run the build in parallel threads.

<<<<<<< HEAD
<<<<<<< HEAD
8. The binary `megamol` is located in the default installation path `../megamol/build/install/bin`.

<<<<<<< HEAD
<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
### Command Line Arguments

Providing additional command line arguments allow individual configuration of global MegaMol behavior and settings.  
<!-- (The command line arguments are only read and interpreted by the `frontend`.)-->

The following command line arguments are available:

**Note:** The *khrdebug* option is currently ignored and not applied.
```
    megamol.exe [OPTION...] <additional project files>

      --host arg      address of lua host server, default:
                      tcp://127.0.0.1:33333
      --example       load minimal test spheres example project
      --khrdebug      enable OpenGL KHR debug messages
      --vsync         enable VSync in OpenGL window
      --window arg    set the window size and position, accepted format:
                      WIDTHxHEIGHT[+POSX+POSY]
      --fullscreen    open maximized window
      --nodecoration  open window without decorations
      --topmost       open window that stays on top of all others
      --nocursor      do not show mouse cursor inside window
      --help          print help
```
=======
    $ make && make install`
>>>>>>> 8e58073aa (...)

<!-- ---------------------------------------------------------------------- -->
### Configuration File

<<<<<<< HEAD
After successfully compiling and installing MegaMol, you should have all executable files inside your `bin` directory (default: `../megamol/build/install/bin`). 
<!-- (The configuration file is only read and interpreted by the MegaMol `core`.) -->
In the `bin` directory, you can find the default configuration file `megamolconfig.lua`:  

```lua
    -- Standard MegaMol Configuration File --
    print("Standard MegaMol Configuration:")

<<<<<<< HEAD
    basePath = "C:/megamol/build/install/"

    mmSetLogLevel("*") -- LogLevel: None=0,Error=1,Warn=100,INFO=200,ALL=*
    mmSetEchoLevel("*")
    -- mmSetLogFile("") 
=======
After successfully installing or compiling MegaMol you should have all executable files inside your bin folder. Some setup still needs to be done.
Create a file `megamolconfig.lua` in this bin directory, with the following content (it may already exist). 
YOU WILL NEED TO ADJUST THE PATHS ACCORDINGLY:
=======
You can append the option `-j 4` to the make command to run the build in 4 parallel threads.
=======
- Use the `-j` option of `make` to run the build in parallel threads.
- The default installation path is `../megamol/build/install/`
>>>>>>> d85984bb9 (docu)
=======
  Use the `-j` option of `make` to run the build in parallel threads.
=======
    Use the `-j` option of `make` to run the build in parallel threads.
>>>>>>> c571ca9c6 (...)
=======
    `$ make && make install`
=======
    $ make && make install
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
=======
```
    $ make && make install
```
>>>>>>> aa0faee09 (docu)

<<<<<<< HEAD
    Hint: Use the `-j` option for `make` to run the build in parallel threads.
>>>>>>> f42be6ab4 (...)
=======
  - Hint: Use the `-j` option for `make` to run the build in parallel threads.
>>>>>>> 9712fbc5e (docu)
=======
    `$ make && make install`

    - Hint: Use the `-j` option for `make` to run the build in parallel threads.
>>>>>>> a3a2b3f57 (docu)

<<<<<<< HEAD
- The default installation path for the binary `megamol` is `../megamol/build/install/bin`
    <!-- TODO Reference shell scipt for use with external libraries like ospray -->
>>>>>>> 88e3119f9 (docu)
=======
- The binary `megamol` is located in the default installation path `../megamol/build/install/bin`.
=======
8. The binary `megamol` is located in the default installation path `../megamol/build/install/bin`.
>>>>>>> 94da8d87e (docu)

  If you use additional external libraries (e.g. when using OSPRay), you have have to use the shell script `megamol.sh` instead. 
  This script adds the required library path:

  ```bash
    #!/bin/bash
    #
    # MegaMol startup script
    # Copyright 2020, https://megamol.org/
    #

    BIN_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
    cd "$BIN_DIR"

    LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH ./megamol "$@"
  ```
>>>>>>> 06b190844 (docu)

=======
>>>>>>> bdc293c78 (docu)
<!-- ---------------------------------------------------------------------- -->
### Command Line Arguments

Providing additional command line arguments allow individual configuration of global MegaMol behaviour and settings.  
<!-- (The command line arguments are only read and interpreted by the `frontend`.)-->

The following command line arguments are available:

**Note:** The *khrdebug* option is currently ignored and not applied.
```
    megamol.exe [OPTION...] <additional project files>

      --host arg      address of lua host server, default:
                      tcp://127.0.0.1:33333
      --example       load minimal test spheres example project
      --khrdebug      enable OpenGL KHR debug messages
      --vsync         enable VSync in OpenGL window
      --window arg    set the window size and position, accepted format:
                      WIDTHxHEIGHT[+POSX+POSY]
      --fullscreen    open maximized window
      --nodecoration  open window without decorations
      --topmost       open window that stays on top of all others
      --nocursor      do not show mouse cursor inside window
      --help          print help
```

<!-- ---------------------------------------------------------------------- -->
### Configuration File

<<<<<<< HEAD
**Note:** Some tagged configuration settings are *DEPRECATED* and ignored!

<<<<<<< HEAD
<<<<<<< HEAD
After successfully compiling and installing MegaMol, you should have all executable files inside your bin folder (default: `../megamol/build/install/`). 
Some setup still needs to be done.
<<<<<<< HEAD
In the `bin` directory, you can find the default configuration file for MegaMol: `megamolconfig.lua`:
>>>>>>> 800f17d5c (manual update)
=======
In the `bin` directory, you can find the default configuration file for MegaMol `megamolconfig.lua`:
>>>>>>> d85984bb9 (docu)
=======
After successfully compiling and installing MegaMol, you should have all executable files inside your bin folder (default: `../megamol/build/install/bin`). 
=======
=======
>>>>>>> 11f6a30ac (docu)
After successfully compiling and installing MegaMol, you should have all executable files inside your `bin` directory (default: `../megamol/build/install/bin`). 
<<<<<<< HEAD
>>>>>>> 8b2342edf (docu)
In the `bin` directory, you can find the default configuration file `megamolconfig.lua`:
<<<<<<< HEAD
>>>>>>> 06b190844 (docu)

=======
>>>>>>> bdc293c78 (docu)
=======
<!-- (The configuration file is only read and interpreted by the MegaMol `core`.) -->
In the `bin` directory, you can find the default configuration file `megamolconfig.lua`:  

>>>>>>> ee6adca4d (docu)
```lua
    -- Standard MegaMol Configuration File --
    print("Standard MegaMol Configuration:")

    basePath = "C:/megamol/build/install/"

    mmSetLogLevel("*") -- LogLevel: None=0,Error=1,Warn=100,INFO=200,ALL=*
    mmSetEchoLevel("*")
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> ba7df3ade (manual update)
=======
    mmSetLogFile("")    
>>>>>>> 06b190844 (docu)
=======
    -- mmSetLogFile("") 
>>>>>>> 2e3d465a0 (...)
    mmSetAppDir(basePath .. "bin")
    mmAddShaderDir(basePath .. "share/shaders")
    mmAddResourceDir(basePath .. "share/shaders")
    mmAddResourceDir(basePath .. "share/resources")
    mmPluginLoaderInfo(basePath .. "bin", "*.mmplg", "include")

    computer = mmGetMachineName()

    mmSetConfigValue("*-window",    "x5y35w1280h720")
    mmSetConfigValue("consolegui",  "on")
    mmSetConfigValue("topmost",     "off")
    mmSetConfigValue("fullscreen",  "off")
    mmSetConfigValue("vsync",       "off")
    mmSetConfigValue("useKHRdebug", "off")
    mmSetConfigValue("arcball",     "off")
```

The following paragraphs explain the essential steps of configuring MegaMol in more detail.

<!-- ---------------------------------------------------------------------- -->
#### General Settings

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
Locate line 3 containing the variable `basePath`. 
Both relative and absolute path should work here fine. 
This path is set automatically and always has to fit the currently used execution path!
<<<<<<< HEAD
```lua
    basePath = "C:/megamol/build/install/"  
=======
Locate line 3 containing the variable `basePath`. Both relative and absolute path should work here fine, it is recommended to change the path in this line to the global path to the MegaMol application directory, e.g.:

```lua
    basePath = "C:/megamol/build/install/"
>>>>>>> ba7df3ade (manual update)
```

<!-- ---------------------------------------------------------------------- -->
#### Logging
=======
Locate line 3 containing the variable `basePath`. Both relative and absolute path should work here fine, **it is necessary to change the path in this line to the global path to the MegaMol application directory**, e.g.:
=======
Locate line 3 containing the variable `basePath`. 
Both relative and absolute path should work here fine. 
<<<<<<< HEAD
This path is set automatically and always has to fit the currently used execution path.
>>>>>>> 06b190844 (docu)
=======
This path is set automatically and always has to fit the currently used execution path!
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)

=======
>>>>>>> bdc293c78 (docu)
```lua
    basePath = "C:/megamol/build/install/"  
```

<!-- ---------------------------------------------------------------------- -->
#### Logging

<<<<<<< HEAD
Line 4-6 configures the logging mechanism of MegaMol . Adjusting the value of *EchoLevel* changes the amount of log information printed on the console. Specifying a log file and the level informs MegaMol to write a log file and print the messages of the requested level into that file. The *LogLevel* is a numeric value. All messages with lower numeric values will be printed (or saved). The asterisk `*` stands for the highest numeric value, thus printing all messages.
>>>>>>> 800f17d5c (manual update)

=======
>>>>>>> 06b190844 (docu)
Line 4-6 configures the logging mechanism of MegaMol. 
Adjusting the value of *EchoLevel* changes the amount of log information printed on the console. 
Specifying a log file and the level informs MegaMol to write a log file and print the messages of the requested level into that file. 
The *LogLevel* is a numeric value. 
All messages with lower numeric values will be printed (or saved). 
The asterisk `*` stands for the highest numeric value, thus printing all messages.
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 06b190844 (docu)
=======
>>>>>>> bdc293c78 (docu)
```lua
    mmSetLogLevel('*') -- LogLevel: None=0,Error=1,Warn=100,INFO=200,ALL=*
    mmSetEchoLevel('*')
<<<<<<< HEAD
<<<<<<< HEAD
    -- mmSetLogFile("") 
```

<!-- ---------------------------------------------------------------------- -->
#### Application, Shaders and Resources
=======
    mmSetLogFile("") 
=======
    -- mmSetLogFile("") 
>>>>>>> 2e3d465a0 (...)
```

<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
#### Application, Shaders and Resources

<<<<<<< HEAD
<<<<<<< HEAD
Line 9+10 define the shader and resource directories:
>>>>>>> 800f17d5c (manual update)
=======
Line 9-11 define the shader and resource directories:
>>>>>>> 06b190844 (docu)
=======
Line 9-11 define the application, shader and resource directories:
>>>>>>> 8b2342edf (docu)
=======
Line 7-10 define the application, shader and resource directories:
<<<<<<< HEAD
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)

Line 7-10 define the application, shader and resource directories:
=======
>>>>>>> bdc293c78 (docu)
```lua
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8b2342edf (docu)
    mmSetAppDir(basePath .. "bin")  
    mmAddShaderDir(basePath .. "share/shaders")
    mmAddResourceDir(basePath .. "share/shaders")
    mmAddResourceDir(basePath .. "share/resources")
=======
    mmAddShaderDir("C:/megamol/build/install/share/shaders")
    mmAddResourceDir("C:/megamol/build/install/share/resources")
>>>>>>> ba7df3ade (manual update)
=======
    mmAddShaderDir(basePath .. "share/shaders")
    mmAddResourceDir(basePath .. "share/shaders")
    mmAddResourceDir(basePath .. "share/resources")
>>>>>>> 800f17d5c (manual update)
```
<<<<<<< HEAD
The *Add...Dir* commands set the paths for the respective resources.

<<<<<<< HEAD
=======
=======
>>>>>>> bdc293c78 (docu)
The *Add...Dir* commands set the paths for the respective resources.

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 800f17d5c (manual update)
<!-- ---------------------------------------------------------------------- -->
=======
[//]: # (----------------------------------------------------------------------) 
>>>>>>> 4fa438626 (manual update ...)
=======
<!-- ---------------------------------------------------------------------- -->
>>>>>>> 0ae2f4429 (manual update ...)
#### Plugins

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
Since we switched to static linking of plugin libraries into the MegaMol binary, the configuration of `mmPluginLoaderInfo` is ***DEPRECATED*** and no longer required.
=======
*DEPRECATED*
=======
*DEPRECATED --- Because static build ... *
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)

<<<<<<< HEAD
Extend the configuration if you introduce new plugins into your installation. Although there are different ways to specify the plugins to be loaded, the tags in the example configuration file are the most secure way. Each `mmPluginLoaderInfo` tag requires three attributes:
>>>>>>> f5eec258f (manual update ....)

<!--
*DEPRECATED:*
=======
Since switched to static linking of plugin libraries into the MegaMol binary, the configuration of *Plugin Loader Info* is no longer required.

<<<<<<< HEAD
***DEPRECATED***

*Mentioned only for legacy purposes:*
>>>>>>> 94da8d87e (docu)
=======
=======
Since switched to static linking of plugin libraries into the MegaMol binary, the configuration of `mmPluginLoaderInfo` is no longer required.
=======
Since switched to static linking of plugin libraries into the MegaMol binary, the configuration of `mmPluginLoaderInfo` is no longer required and ***DEPRECATED***.
>>>>>>> 11f6a30ac (docu)
=======
Since we switched to static linking of plugin libraries into the MegaMol binary, the configuration of `mmPluginLoaderInfo` is ***DEPRECATED*** and no longer required.
>>>>>>> ee6adca4d (docu)

<!--
>>>>>>> c1b6286c7 (docu)
*DEPRECATED:*
>>>>>>> ce3f629ba (docu)

Extend the configuration if you introduce new plugins into your installation. 
Although there are different ways to specify the plugins to be loaded, the tags in the example configuration file are the most secure way. 
Each `mmPluginLoaderInfo` tag requires three attributes:
=======
Extend the configuration if you introduce new plugins into your installation. 
Although there are different ways to specify the plugins to be loaded, the tags in the example configuration file are the most secure way. 
Each `mmPluginLoaderInfo` tag requires three attributes:

- `path` should be the path to find the plugin. The example configuration file assumes to find the plugins in the same directory as the MegaMol executable (which is the
case for Windows installations. 
  On Linux systems, you need to change this path, e.g. to `../../lib/megamol`.
- `name` is the file name of the plugin.
- `action` refers to an internal parameter of MegaMol and should always be `include`.

Rendering modules from plugins require shader codes to function. 
MegaMol searches these codes in all registered shader directories. 
To register a shader directory, add a corresponding tag to the configuration file.
<<<<<<< HEAD
>>>>>>> 06b190844 (docu)

- `path` should be the path to find the plugin. The example configuration file assumes to find the plugins in the same directory as the MegaMol executable (which is the
case for Windows installations. 
  On Linux systems, you need to change this path, e.g. to `../../lib/megamol`.
- `name` is the file name of the plugin.
- `action` refers to an internal parameter of MegaMol and should always be `include`.

Rendering modules from plugins require shader codes to function. 
MegaMol searches these codes in all registered shader directories. 
To register a shader directory, add a corresponding tag to the configuration file.
=======
>>>>>>> bdc293c78 (docu)
```lua
    mmPluginLoaderInfo(basePath .. "bin", "*.mmplg", "include")
```
-->

<!-- ---------------------------------------------------------------------- -->
#### Global Settings

<<<<<<< HEAD
<<<<<<< HEAD
The configuration file also specifies global settings variables which can modify the behavior of different modules.
=======
The configuration file also specifies global settings variables which can modify the behavior of different modules. 
Two such variables are set in the example configuration file.
In line 14 the variable `*-window` is set. 
=======
The configuration file also specifies global settings variables which can modify the behavior of different modules.

- The following settings variable activates (or deactivates) the *arcball* camera behavior. Set this option to `on` in order to use the *arcball* camera navigation.
```lua
    mmSetConfigValue("arcball",     "off")
```

All other configuration options are ***DEPRECATED*** and have currently no effect!

<!--
*DEPRECATED:*

<<<<<<< HEAD
<<<<<<< HEAD
For example, in line 14 the variable `*-window` is set. 
>>>>>>> 94da8d87e (docu)
This variable specifies the default position and size for all rendering windows MegaMol will create. 
The asterisk represents any window name. 
If you set a variable with a specific name, windows with exactly this name will respect the settings variable. 
For example, `test-window` will specify the value for the window created by the view instance test.
<<<<<<< HEAD
>>>>>>> 06b190844 (docu)
=======
The value itself contains five variables:
- The first two variables are prefixed with `x` and `y` and specify the location of the window in screen pixel coordinates.
- The second two variables are prefixed with `w` and `h` and specify the size of the client area of the window in pixels.
- The last optional variable `nd` (stands for **n**o **d**ecorations) will remove all window decorations, buttons, and border from the created window. 
This variable allows us to create borderless windows filling the complete screen for full-screen rendering.
>>>>>>> 94da8d87e (docu)

- The following settings variable activates (or deactivates) the *arcball* (orbiting) camera behavior. Set this option to `on` in order to use the *arcball* camera navigation.
```lua
<<<<<<< HEAD
    mmSetConfigValue("arcball",     "off")
=======
    mmSetConfigValue("*-window",    "x5y35w1280h720")
>>>>>>> 800f17d5c (manual update)
```
=======
- For example, in line 14 the variable `*-window` is set. 
=======
- In line 14 the variable `*-window` is set. 
>>>>>>> c1b6286c7 (docu)
    This variable specifies the default position and size for all rendering windows MegaMol will create. 
    The asterisk represents any window name. 
    If you set a variable with a specific name, windows with exactly this name will respect the settings variable. 
    For example, `test-window` will specify the value for the window created by the view instance test.
    The value itself contains five variables:
    - The first two variables are prefixed with `x` and `y` and specify the location of the window in screen pixel coordinates.
    - The second two variables are prefixed with `w` and `h` and specify the size of the client area of the window in pixels.
    - The last optional variable `nd` (stands for **n**o **d**ecorations) will remove all window decorations, buttons, and border from the created window. 
    This variable allows us to create borderless windows filling the complete screen for full-screen rendering.
<<<<<<< HEAD
>>>>>>> ce3f629ba (docu)

<<<<<<< HEAD
All other configuration options are ***DEPRECATED*** and have currently no effect!

<!--
*DEPRECATED:*

- In line 14 the variable `*-window` is set. 
    This variable specifies the default position and size for all rendering windows MegaMol will create. 
    The asterisk represents any window name. 
    If you set a variable with a specific name, windows with exactly this name will respect the settings variable. 
    For example, `test-window` will specify the value for the window created by the view instance test.
    The value itself contains five variables:
    - The first two variables are prefixed with `x` and `y` and specify the location of the window in screen pixel coordinates.
    - The second two variables are prefixed with `w` and `h` and specify the size of the client area of the window in pixels.
    - The last optional variable `nd` (stands for **n**o **d**ecorations) will remove all window decorations, buttons, and border from the created window. 
    This variable allows us to create borderless windows filling the complete screen for full-screen rendering.
=======
>>>>>>> bdc293c78 (docu)
```lua
    mmSetConfigValue("*-window",    "x5y35w1280h720")
```

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
*DEPRECATED:*
=======
The last settings variable, activates (or deactivates) the the arcball camera behaviour. Set this option to `on` in order to use the arcball camera navigation.
<<<<<<< HEAD
<!-- TODO: Needs more explanation of the available hotkeys! -->
>>>>>>> 800f17d5c (manual update)
=======
>>>>>>> f5eec258f (manual update ....)
=======
The last settings variable, activates (or deactivates) the arcball camera behavior. Set this option to `on` in order to use the arcball camera navigation.
>>>>>>> 0ae2f4429 (manual update ...)
=======
The last settings variable activates (or deactivates) the arcball camera behavior. Set this option to `on` in order to use the arcball camera navigation.
>>>>>>> 06b190844 (docu)
=======
***DEPRECATED***
=======
*DEPRECATED:*
>>>>>>> ce3f629ba (docu)

- This variable defines whether the GUI is show or not.
```lua
    mmSetConfigValue("consolegui",  "on")
```

*DEPRECATED:*

- Show MegaMol window on top of other windows or not.
```lua    
    mmSetConfigValue("topmost",     "off")
```

*DEPRECATED:*

- Show MegMol window in fullscreen or not.
```lua    
    mmSetConfigValue("fullscreen",  "off")
```

*DEPRECATED:*

- Enable or disable VSync (vertical synchronization).
```lua    
    mmSetConfigValue("vsync",       "off")
```

*DEPRECATED:*

- Defines wether the OpenGL Debug Output (KHR extension)[https://www.khronos.org/opengl/wiki/Debug_Output] is used or not.
```lua    
    mmSetConfigValue("useKHRdebug", "off")
```

<<<<<<< HEAD
<<<<<<< HEAD
The last settings variable activates (or deactivates) the *arcball* camera behavior. Set this option to `on` in order to use the *arcball* camera navigation.
>>>>>>> 94da8d87e (docu)
=======
- The last settings variable activates (or deactivates) the *arcball* camera behavior. Set this option to `on` in order to use the *arcball* camera navigation.
>>>>>>> ce3f629ba (docu)

- This variable defines whether the GUI is show or not.
```lua
<<<<<<< HEAD
    mmSetConfigValue("consolegui",  "on")
=======
    mmSetConfigValue("arcball",     "off")
>>>>>>> 800f17d5c (manual update)
```

<<<<<<< HEAD
<<<<<<< HEAD
*DEPRECATED:*

<<<<<<< HEAD
<<<<<<< HEAD
- Show MegaMol window on top of other windows or not.
```lua    
    mmSetConfigValue("topmost",     "off")
```
=======
<!-- ---------------------------------------------------------------------- -->
<a name="tests"></a>
>>>>>>> 800f17d5c (manual update)

*DEPRECATED:*
=======
[//]: # (----------------------------------------------------------------------) 
=======
This concludes the building and configuring of MegaMol.
=======
This concludes the information on building and the options on how to configure MegaMol.
>>>>>>> 94da8d87e (docu)
Test your installation following the description in the following section.

<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 0ae2f4429 (manual update ...)
### Tests
>>>>>>> 4fa438626 (manual update ...)
=======
### Test Installation
>>>>>>> 9f08ae8ac (updted cineamtic and ospray docu)
=======
=======

<!-- ###################################################################### -->
-----
>>>>>>> 83aa4eadb (docu)
## Test Installation
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)

<<<<<<< HEAD
<<<<<<< HEAD
- Show MegMol window in full screen mode or not.
```lua    
    mmSetConfigValue("fullscreen",  "off")
```
=======
To test MegaMol, simply start the frontend executable. Open a console and change your working directory to the MegaMol install directory. Start
the MegaMol start script:
>>>>>>> 800f17d5c (manual update)
=======
*REWORK*

In order to test the installtion, simply start the frontend executable. 
Open a console and change your working directory to the MegaMol install directory (default: `../megamol/build/install/bin`). 
Execute the MegaMol binary:
>>>>>>> 06b190844 (docu)

*DEPRECATED:*

- Enable or disable VSync (vertical synchronization).
```lua    
    mmSetConfigValue("vsync",       "off")
```

*DEPRECATED:*

<<<<<<< HEAD
- Defines wether the OpenGL Debug Output (KHR extension)[https://www.khronos.org/opengl/wiki/Debug_Output] is used or not.
```lua    
    mmSetConfigValue("useKHRdebug", "off")
```
--> 

=======
>>>>>>> bdc293c78 (docu)
This concludes the information on building and the options on how to configure MegaMol.
Test your installation following the description in the following section.
=======
    > megamol.exe

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<!-- ???
The resulting output should look something like this:
>>>>>>> 800f17d5c (manual update)


<<<<<<< HEAD
<!-- ###################################################################### -->
-----
## Test Installation

In order to test the installation, simply run the frontend executable.  
Open a console (e.g *Linux Terminal* or *Windows Powershell*) and change your working directory to the MegaMol install directory containing the `bin` folder (default: `../megamol/build/install/bin`) and execute the MegaMol binary:
=======
        MegaMol� Console
        Copyright (c) 2006 - 2017 by MegaMol Team: VISUS (Universitaet Stuttgart, Germany), TU Dresden (Dresden, Germany)
        Alle Rechte vorbehalten.
        All rights reserved.
        
    0200|Called: /nethome/user/software/megamol/bin/mmconsole
    250|Path "/nethome/user/software/megamol/share/shaders" added as shader search path.
    250|Configuration value "*-window" set to "w1280h720".
    250|Configuration value "consolegui" set to "on".
    350|Directory "application" is "/nethome/user/software/megamol/bin"
    200|Configuration sucessfully loaded from "/nethome/user/software/megamol/bin/megamol.cfg"
    200|Default LRHostAddress = "tcp://*:33333"
    200|Default LRHostEnable = "true"
    200|Installed service "LuaRemote" [1]
    200|LRH Server socket opened on "tcp://*:33333"
    200|Auto-enabled service "LuaRemote" [1]
    200|Plugin CinematicCamera loaded: 4 Modules, 1 Calls
    200|Plugin "CinematicCamera" (/nethome/user/software/megamol/lib/libcinematiccamera.mmplg) loaded: 4 Modules, 1 Calls registered
    200|Plugin infovis loaded: 3 Modules, 2 Calls
    200|Plugin "infovis" (/nethome/user/software/megamol/lib/libinfovis.mmplg) loaded: 3 Modules, 2 Calls registered
    200|Plugin mdao2 loaded: 1 Modules, 0 Calls
    200|Plugin "mdao2" (/nethome/user/software/megamol/lib/libmdao2.mmplg) loaded: 1 Modules, 0 Calls registered
    200|Plugin mmstd_datatools loaded: 43 Modules, 4 Calls
    200|Plugin "mmstd_datatools" (/nethome/user/software/megamol/lib/libmmstd_datatools.mmplg) loaded: 43 Modules, 4 Calls registered
    200|Plugin mmstd_moldyn loaded: 16 Modules, 1 Calls
    200|Plugin "mmstd_moldyn" (/nethome/user/software/megamol/lib/libmmstd_moldyn.mmplg) loaded: 16 Modules, 1 Calls registered
    200|Plugin mmstd_trisoup loaded: 12 Modules, 4 Calls
    200|Plugin "mmstd_trisoup" (/nethome/user/software/megamol/lib/libmmstd_trisoup.mmplg) loaded: 12 Modules, 4 Calls registered
    200|Plugin "mmstd.volume" (/nethome/user/software/megamol/lib/libmmstd_volume.mmplg) loaded: 7 Modules, 0 Calls registered
    200|Plugin Protein loaded: 55 Modules, 9 Calls
    200|Plugin "Protein" (/nethome/user/software/megamol/lib/libprotein.mmplg) loaded: 55 Modules, 9 Calls registered
    200|Plugin Protein_Calls loaded: 0 Modules, 10 Calls
    200|Plugin "Protein_Calls" (/nethome/user/software/megamol/lib/libprotein_calls.mmplg) loaded: 0 Modules, 10 Calls registered
    200|Core Instance destroyed
    200|LRH Server socket closed
-->
=======
Alternatively, you can descend into the bin directory and start the frontend directly. 
=======
Alternatively, you can descend into the `bin` directory and start the frontend directly. 
>>>>>>> 8b2342edf (docu)
Doing so, you must ensure that the additional shared objects can be found and loaded. 
Enter the commands. 
To test this, try:

    cd bin
    LD_LIBRARY_PATH=../lib ./megamol

This direct invocation is not recommended. 
Thus, the remaining examples in this manual will assume that you use the start shell script. 
MegaMol should start and print several messages to the console. 
The leading number of each line is the log level. 

<!-- ---------------------------------------------------------------------- -->
### Examples

The [example project script files](https://github.com/UniStuttgart-VISUS/megamol-examples) are automatically available in the `examples` directory, which is installed next to the `bin` directory.

For a better test, you should invoke MegaMol loading an example project script requesting a simple rendering. 
Then you can be sure that the graphics drivers, graphics libraries, and shader codes are correctly found and are working. 
To do this, try: 
>>>>>>> 06b190844 (docu)

=======
>>>>>>> f5eec258f (manual update ....)
Alternatively, you can descend into the bin directory and start the frontend directly. Doing so, you must ensure that the additional shared objects can be found and loaded. Enter the commands. To test this, try:
>>>>>>> 800f17d5c (manual update)

**Windows:**
```
    > megamol.exe
```

**Linux:**

```
    $ ./megamol
```

<<<<<<< HEAD
If additional external libraries are required (e.g. when using the OSPRay plugin), for Linux you have to run the provided shell script `./megamol.sh` instead. 
This script adds the required library path:
=======
    $ ./megamol.sh ../examples/testspheres_megamol.lua
>>>>>>> 800f17d5c (manual update)

```bash
    #!/bin/bash
    #
    # MegaMol startup script
    # Copyright 2020, https://megamol.org/
    #

<<<<<<< HEAD
    BIN_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
    cd "$BIN_DIR"
=======
    > megamol.exe ..\examples\testspheres_megamol.lua
>>>>>>> 800f17d5c (manual update)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH ./megamol "$@"
```
=======
![Test Project](pics/testspheres.png)
<<<<<<< HEAD
Screenshot MegaMol running the test spheres instance. 
The highlighted option in the AntTweak-Bar on the right side of the window adjusts the animation speed.
>>>>>>> f5eec258f (manual update ....)

<<<<<<< HEAD
<<<<<<< HEAD
MegaMol should start and print several messages to the console and an empty rendering window should appear.
You can either check the console log messages or the messages printed in the *Log Console* window.
The leading number of each line is the log level.
There should be no error messages (log level **1**). 
Some warnings (log level **100**) might occur but are *normal* and indicate no failed installation or execution.  

<!-- ---------------------------------------------------------------------- -->
### Examples
=======
<!-- XXX Do not mind the `Ignoring Xlib error: error code n request code m` messages. -->
=======
[//]: # (XXX Do not mind the `Ignoring Xlib error: error code n request code m` messages.) 
>>>>>>> 4fa438626 (manual update ...)
=======
Screenshot of MegaMol running the test spheres instance. 
In the highlighted parameter group `anim` in the *ImGui* on the left side of the window you can adjust the animation speed.

<!-- XXX Do not mind the `Ignoring Xlib error: error code n request code m` messages. -->
>>>>>>> 0ae2f4429 (manual update ...)

=======
In the highlighted parameter group `anim` in the *ImGui* on the left side of the window you can adjust the animation speed.
>>>>>>> 8b2342edf (docu)
=======
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
MegaMol should now open a rendering window showing a generated dataset with several colored spheres. 
Hitting the `space` key starts and stops the animation playback.
In the *Parameters* window you can find all available parameters of the running MegaMol instance grouped by the modules.
For example, you can find the parameter `speed` in the group `inst::view::anim`. 
With this parameter, you can adjust the playback speed of the animation.
In the parameter group `anim` of the `view` module you can adjust the animation speed.

![Test Project](pics/testspheres.png)
*Screenshot of MegaMol running the test spheres instance.*


<!-- ###################################################################### -->
-----
## Project Files

*TODO*

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
A detailed description of the GUI and the configurator can be found in the readme file of the [GUI plugin](https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/gui/README.md).


<<<<<<< HEAD
<<<<<<< HEAD
<!-- ###################################################################### -->
<a name="viewing-data-sets"></a>
>>>>>>> 800f17d5c (manual update)

The [example project script files](https://github.com/UniStuttgart-VISUS/megamol-examples) are automatically available in the `examples` directory, which is installed next to the `bin` directory.

<<<<<<< HEAD
<<<<<<< HEAD
For a better test, you should invoke MegaMol loading an example project script requesting a simple rendering. 
Then you can be sure that the graphics drivers, graphics libraries, and shader codes are correctly found and are working. 
To do this, try: 
=======
## Viewing Data Sets
<!-- XXX More suitable caption name? -->
>>>>>>> f5eec258f (manual update ....)
=======
[//]: # (######################################################################) 
## Viewing Data Sets
[//]: # (XXX More suitable caption name?) 
>>>>>>> 4fa438626 (manual update ...)
=======
<!-- ###################################################################### -->
-----
## Viewing Data Sets
<!-- XXX More suitable caption name? -->
>>>>>>> 0ae2f4429 (manual update ...)

**Linux:**
=======
In this chapter, we discuss the principle usage of the prepared project files for data set viewing. 
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<!-- DEPRECATED This project script files are available in the *script and example* package from the MegaMol project website. -->

<!-- ---------------------------------------------------------------------- -->
<a name="modules-views-calls"></a>
>>>>>>> 800f17d5c (manual update)

    $ ./megamol ../examples/testspheres_megamol.lua
=======
[//]: # (DEPRECATED This project script files are available in the *script and example* package from the MegaMol project website.) 
=======
<!-- DEPRECATED This project script files are available in the *script and example* package from the MegaMol project website. -->
>>>>>>> 0ae2f4429 (manual update ...)
=======
This project script files are available in the `examples` directory, which is installed next to the `bin` directory.
>>>>>>> 8b2342edf (docu)

<!-- ---------------------------------------------------------------------- -->
### Modules, Views and Calls
>>>>>>> 4fa438626 (manual update ...)

<<<<<<< HEAD
<<<<<<< HEAD
**Windows:** 
=======
The runtime functionality of MegaMol is constructed by *modules* and *calls*. These two type of objects are instantiated at runtime, interconnected and build the *module graph*. The figure [Example Graph](#examplegraph) shows an example module graph containing a rendering content of a window *view*, a *renderer*, a *data source*, and two modules providing additional information for the renderer. The modules, shown as blue boxes, are interconnected by *call* objects, shown as gray boxes. The connection endpoints at the modules are *CallerSlots* (outgoing, located on the right of modules) or *CalleeSlots* (incoming, located on the left side of modules) shown as circles.
>>>>>>> 800f17d5c (manual update)

<<<<<<< HEAD
    > megamol.exe ..\examples\testspheres_megamol.lua
=======
![Example Graph](pics/example_graph.png)
<<<<<<< HEAD
An example module graph. Left-most module view of class View3D represents the rendering content of a window. The center module renderer of class SphererRenderer is called by the window using the corresponding call of type CallRenderer3D. The right modules provide data and additional information for the renderer, namely a color map function and a clip plane.An example module graph.
<<<<<<< HEAD
>>>>>>> f5eec258f (manual update ....)

MegaMol should now open a rendering window showing a generated dataset with several colored spheres and the outline of the bounding box. 
Hitting the `space` key starts and stops the animation playback.
In the GUI window *Parameters* you can find all available parameters of the running MegaMol instance grouped by the modules.
For example, you can find the parameter `speed` in the group `inst::view::anim`. 
With this parameter, you can adjust the playback speed of the animation.
In the parameter group `anim` of the `view` module you can adjust the animation speed.  

<<<<<<< HEAD
Alternatively, you can also open an empty MegaMol rendering window and load the above example project script file via the menu `File / Load Project`.  
=======
<!-- ---------------------------------------------------------------------- -->
=======

The module graph follows the pull pattern. This means that modules request function invocation by other modules. For example, the *view* module needs to update the window content. The *view* module thus invokes the *renderer* module to provide a new rendering. The *renderer* calls the data source if new data is available or to provide the old cached data.

[//]: # (----------------------------------------------------------------------) 
>>>>>>> 4fa438626 (manual update ...)
=======
An example module graph. Left-most module view of class View3D represents the rendering content of a window. The center module renderer of class SphererRenderer is called by the window using the corresponding call of type CallRenderer3D. The right modules provide data and additional information for the renderer, namely a color map function and a clip plane. An example module graph.
=======
The runtime functionality of MegaMol is constructed by *modules* and *calls*. 
These two type of objects are instantiated at runtime, interconnected and build the *module graph*. 
The figure [Example Graph](#examplegraph) shows an example module graph containing a rendering content of a window *view*, a *renderer*, a *data source*, and two modules providing additional information for the renderer. 
The modules, shown as blue boxes, are interconnected by *call* objects, shown as gray boxes. 
The connection endpoints at the modules are *CallerSlots* (outgoing, located on the right of modules) or *CalleeSlots* (incoming, located on the left side of modules) shown as circled dots.

The module graph follows the pull pattern. 
This means that modules request function invocation by other modules. 
For example, the *view* module needs to update the window content. 
The *view* module thus invokes the *renderer* module to provide a new rendering. 
The *renderer* calls the data source if new data is available or to provide the old cached data.
>>>>>>> 8b2342edf (docu)

Left-most module view of class `View3D_2` represents the rendering content of a window. 
The center module renderer of class `SphererRenderer` is called by the window using the corresponding call of type `CallRenderer3D`. 
The right modules provide data and additional information for the renderer, namely a color map function and a clip plane. 

![Example Graph](pics/example_graph.png)
*An example module graph.*

<!-- ---------------------------------------------------------------------- -->
<<<<<<< HEAD
>>>>>>> 0ae2f4429 (manual update ...)
#### Modules and calls
>>>>>>> 800f17d5c (manual update)

<<<<<<< HEAD
All available options provided via the graphical user interface are described separately in the readme file of the [GUI plugin](../plugins/gui).
=======
#### Modules and Calls
>>>>>>> 6668c26ff (docu toc)
=======
- Defines wether the OpenGL Debug Output (KHR extension)[https://www.khronos.org/opengl/wiki/Debug_Output] is used or not.
```lua    
    mmSetConfigValue("useKHRdebug", "off")
```
--> 
>>>>>>> c1b6286c7 (docu)

![Test Project](pics/testspheres.png)
*Screenshot of MegaMol running the test spheres instance.*

<<<<<<< HEAD
<<<<<<< HEAD

<!-- ###################################################################### -->
-----
<<<<<<< HEAD
## MegaMol Graph

In this chapter, we show the operating principle of MegaMol which is required to creating own custom projects for MegaMol.
=======
*Modules* are the functional entities of MegaMol. They provide several programmatic access points, the *slots*. Two types of these slots are shown in figure [Example Graph](#examplegraph) as colored arrowheads.

*CalleeSlots* are access points of modules, through which these can be called to perform a function. For example, modules of class `SphererRenderer` provide a CalleeSlot rendering through which the rendering function can be invoked. The counterparts are CallerSlots which are outgoing access points. These allow modules to call other modules. Modules of class `View3D` provide a corresponding slot `rendering` to call a connected renderer.
These two types of slots are connected using objects of *call* classes. These are shown as gray boxes in figure [Example Graph](#examplegraph). Both *CalleeSlots* and *CallerSlots* specify types of calls they are compatible with. In the case of the above examples of renderings-relates slots, this is the type `CallRender3D`.

*Calls* should be lightweight. Instead, they are thin interfaces meant for data transport. For example, data to be visualized is loaded by data source modules. In [Example Graph](#examplegraph) the module *data* of class *MMPLDDataSource* loads a specified data set into main memory and provides the data through its  CalleeSlot*. The data is accessed through a *MultiParticleDataCall*. The call, however, does not copy the data but provides access to the data in terms of memory pointers, and metadata. This avoidance of copy operations is most important and one of the core design ideas of MegaMol.
>>>>>>> 0ae2f4429 (manual update ...)

<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
<<<<<<< HEAD
### Modules, Views and Calls
=======
## Test Installation

In order to test the installtion, simply run the frontend executable.  
Open a console (e.g *Linux Terminal* or *Windows Powershell*) and change your working directory to the MegaMol  install directory containing the `bin` folder (default: `../megamol/build/install/bin`) and execute the MegaMol binary:

<<<<<<< HEAD
Windows:
>>>>>>> bdc293c78 (docu)
=======
**Windows:**
<<<<<<< HEAD
>>>>>>> c1b6286c7 (docu)

The runtime functionality of MegaMol is constructed by *modules* and *calls*. 
These two types of objects are instantiated at runtime, interconnected and build the *module call graph*. 
The figure given below, shows an example module call graph containing a *view*, the rendering content of a window, a *renderer*, a *data source*, and some modules providing additional information for the renderer. 
The modules are interconnected by the *call* objects. 
The connection endpoints at the modules are *CallerSlots* (outgoing, located on the right side of modules) or *CalleeSlots* (incoming, located on the left side of modules) shown as circled dots.
=======
<a name=views></a>

<<<<<<< HEAD
=======
[//]: # (----------------------------------------------------------------------) 
>>>>>>> 4fa438626 (manual update ...)
#### Views 

<<<<<<< HEAD
*Views* are one of the two instance types MegaMol can run. They are specified by the corresponding tag in a MegaMol project file (see section [Project Files](#project-files)). When a view is instantiated, a corresponding namespace will be created, and all modules instantiated as part of the view will be created inside this namespace. For example, the project file seen in next section ([Project Files](#project-files)) defines the module data as part of the view dataview. If this view is instantiated by the command line note (note that files might need some path adjustments):
>>>>>>> 800f17d5c (manual update)

The module call graph follows the pull pattern. 
This means that modules request function invocation by other modules. 
For example, the *view* module needs to update the window content. 
The *view* module thus invokes the *renderer* module to provide a new rendering. 
The *renderer* calls the data source if new data is available or to provide the old cached data.
=======
```
    > megamol.exe
```
>>>>>>> 88d2d64e8 (docu)

<<<<<<< HEAD
Left-most module view of class `View3D_2` represents the rendering content of a window. 
The center module renderer of class `BoundingBoxRenderer` and `SphererRenderer` are called subsequently by the window using the corresponding call of type `CallRenderer3D_2`. 
The right modules provide data and additional information for the renderer, namely a color map transfer function and a clipping plane. 
=======
Linux:
=======
**Linux:**
>>>>>>> c1b6286c7 (docu)

```
    $ ./megamol
```

If additional external libraries are required (e.g. when using the OSPRay plugin), for Linux you have have to run the provided shell script `./megamol.sh` instead. 
This script adds the required library path:

```bash
    #!/bin/bash
    #
    # MegaMol startup script
    # Copyright 2020, https://megamol.org/
    #

    BIN_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
    cd "$BIN_DIR"

    LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH ./megamol "$@"
```

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 11f6a30ac (docu)
MegaMol should start and print several messages to the console and an empty rendering window should appear.
You can either check the console log messages or the messages printed in the *Log Console* window.
The leading number of each line is the log level.
There should be no error messages (log level **1**). 
<<<<<<< HEAD
<<<<<<< HEAD
Some warnings (log level **100**) might occur but indicate no failed installation or execution.
>>>>>>> bdc293c78 (docu)
=======
Some warnings (log level **100**) might occur but are *normal* and indicate no failed installation or execution.
>>>>>>> c1b6286c7 (docu)

<<<<<<< HEAD
<<<<<<< HEAD
![Example Graph](pics/example_graph.png)  
=======
=======
>>>>>>> bc7ad0b5d (docu)
=======
Some warnings (log level **100**) might occur but are *normal* and indicate no failed installation or execution.  

>>>>>>> 11f6a30ac (docu)
<!-- ---------------------------------------------------------------------- -->
<a name="project-files"></a>
>>>>>>> 800f17d5c (manual update)
=======
*Modules* are the functional entities of MegaMol. 
They provide several programmatic access points, the *slots*. 
Two types of these slots are shown in figure [Example Graph](#examplegraph) as circled dots.

<<<<<<< HEAD
*CalleeSlots* are access points of modules, through which these can be called to perform a function. 
For example, modules of class `SphererRenderer` provide a CalleeSlot rendering through which the rendering function can be invoked. 
The counterparts are CallerSlots which are outgoing access points. 
These allow modules to call other modules. 
Modules of class `View3D_2` provide a corresponding slot `rendering` to call a connected renderer.
These two types of slots are connected using objects of *call* classes. 
These are shown as gray boxes in figure [Example Graph](#examplegraph). 
Both *CalleeSlots* and *CallerSlots* specify types of calls they are compatible with. 
In the case of the above examples of renderings-relates slots, this is the type `CallRender3D`.
=======
**Linux:**
>>>>>>> c1b6286c7 (docu)

<<<<<<< HEAD
*Calls* should be lightweight. 
Instead, they are thin interfaces meant for data transport. 
For example, data to be visualized is loaded by data source modules. 
In [Example Graph](#examplegraph) the module *data* of class *MMPLDDataSource* loads a specified data set into main memory and provides the data through its  CalleeSlot*. 
The data is accessed through a *MultiParticleDataCall*. 
The call, however, does not copy the data but provides access to the data in terms of memory pointers, and metadata. 
This avoidance of copy operations is most important and one of the core design ideas of MegaMol.
=======
    $ ./megamol ../examples/testspheres_megamol.lua
>>>>>>> bdc293c78 (docu)

*Parameter slots* are the third type of slots. 
These are access points to exposed parameters controlling the functionality. 
Such parameters are automatically included in the frontend’s GUI. 
Examples of such parameters are the setup of the virtual camera and light source in modules of type `View3D` or the dataset file name in data source modules.  

<<<<<<< HEAD
The *module graph* is configured for MegaMol using a project file. 
These files define modules and interconnecting calls for different instance specifications. 
There are two types of instances:
Views (see section [Views](#views)) and jobs (see section [Jobs](#jobs)). 
The starting command line of the console front-end loads project files (using `-p`) and requests instantiation of views and jobs (using `-i`).
>>>>>>> 8b2342edf (docu)
=======
**Windows:** 
>>>>>>> c1b6286c7 (docu)

<<<<<<< HEAD
*Example module call graph.*
=======
[//]: # (----------------------------------------------------------------------) 
=======
<!-- ---------------------------------------------------------------------- -->
#### Views 

<<<<<<< HEAD
*UPDATE/DEPRECATED*
=======
You can also open an empty MegaMol rendering window and load the example project file via the menu.


=======
>>>>>>> a08d49950 (docu)
MegaMol should now open a rendering window showing a generated dataset with several colored spheres and the outline of the bounding box. 
Hitting the `space` key starts and stops the animation playback.
In the GUI window *Parameters* you can find all available parameters of the running MegaMol instance grouped by the modules.
For example, you can find the parameter `speed` in the group `inst::view::anim`. 
With this parameter, you can adjust the playback speed of the animation.
In the parameter group `anim` of the `view` module you can adjust the animation speed.  

Alternatively, you can also open an empty MegaMol rendering window and load the above example project script file via the menu `File / Load Project`.  

All available options provided via the graphical user interface are described separately in the readme file of the [GUI plugin](../plugins/gui).

![Test Project](pics/testspheres.png)
*Screenshot of MegaMol running the test spheres instance.*
>>>>>>> bdc293c78 (docu)

*Views* are one of the two instance types MegaMol can run. 
They are specified by the corresponding tag in a MegaMol project file (see section [Project Files](#project-files)). 
When a view is instantiated, a corresponding namespace will be created, and all modules instantiated as part of the view will be created inside this namespace. 
For example, the project file seen in next section ([Project Files](#project-files)) defines the module data as part of the view dataview. 
If this view is instantiated by the command line note (note that files might need some path adjustments):

<<<<<<< HEAD
    $ ./megamol.sh -p ../docs/samples/projects/pdbcartoonview.mmprj -i pdbcartoonview pv --paramfile ../docs/samples/projects/pdbmolview02.param -v ::pdbdata::pdbFilename ../docs/samples/sampledata/1m40_sim.pdb -v ::pdbdata::xtcFilename ../docs/samples/sampledata/1m40_100frames.xtc

Then the module will be created with the full name `::inst::data`. 
Correspondingly, its parameter slot `filename` can be globally addressed by `::inst::data::filename`. 
This allows for the instantiation of several independent view instances. 
For each view instance, a rendering window will be created. 
To provide the content for the rendering window, each view instance description needs to provide a *default view*, usually via the `viewmod` attribute of the view tag in the MegaMol project file. 
The value of this attribute is the name for the view module to be called by the window management code. 
This module class must be implemented by deriving from `::megamol::core::view::AbstractView`. 
Typically, you use `View3D` or `View2D`.
MegaMol provides some internal description of views which can be instantiated without loading a project file first. 
The view description *TestSpheres* used in section [Test](#tests) is one example of such a built-in description.

<!-- ---------------------------------------------------------------------- -->
>>>>>>> 0ae2f4429 (manual update ...)
### Project Files
>>>>>>> 4fa438626 (manual update ...)
=======
A detailed description of the GUI and the configurator can be found in the readme file of the [GUI plugin](plugins/gui#2-configurator).
=======
A detailed description of the GUI and the configurator can be found in the readme file of the [GUI plugin](../plugins/gui#2-configurator).
>>>>>>> 83aa4eadb (docu)
=======
A detailed description of the GUI and the configurator can be found in the readme file of the [GUI plugin](../plugins/gui#configurator).
>>>>>>> 98308507c (docu)
Start project by adding entry module `View3D_2` ...
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
#### Modules and Calls
=======
=======
*TODO*
=======
*TODO/UPDATE*

Project files are `lua`scripts.
>>>>>>> 8b2342edf (docu)

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f5eec258f (manual update ....)
<!-- COMPLETE REWORK / DEPRECTED => lua
=======
[//]: # (COMPLETE REWORK / DEPRECTED => lua
>>>>>>> 4fa438626 (manual update ...)

Project files are the primary method to start up MegaMol. The snippets below show the content of the project file `simple_siff.mmprj` and `pdbcartoonview.mmprj` which can be used to view the sample particle datasets.
>>>>>>> 800f17d5c (manual update)

*Modules* are the functional entities of MegaMol. 
They provide several programmatic access points, the *slots*. 
Two types of these slots are shown in the above figure of an example graph as circled dots.

<<<<<<< HEAD
*CalleeSlots* are access points of modules, through which these can be called to perform a function. 
For example, modules of class `SphererRenderer` provide a CalleeSlot `rendering` through which the rendering function can be invoked. 
The counterparts are CallerSlots which are outgoing access points. 
These allow modules to call other modules. 
View modules of class `View3D_2` provide a corresponding slot `rendering` to call a connected renderer.
These two types of slots are connected using objects of *call* classes. 
Both *CalleeSlots* and *CallerSlots* specify types of calls they are compatible with. 
In the case of the above examples of renderings-relates slots, this is the type `CallRender3D_2`.

*Calls* should be lightweight. 
They are thin interfaces meant for data transport. 
For example, data to be visualized is loaded by data source modules. 
In the example graph figure the module *data* of class *MMPLDDataSource* loads a specified data set into main memory and provides the data through its *CalleeSlot*. 
The data is accessed through a *MultiParticleDataCall*. 
The call, however, does not copy the data but provides access to the data in terms of memory pointers, and metadata. 
This avoidance of copy operations is most important and one of the core design ideas of MegaMol.
=======
=======
<!-- COMPLETE REWORK / DEPRECTED => lua

Project files are the primary method to start up MegaMol. The snippets below show the content of the project file `simple_siff.mmprj` and `pdbcartoonview.mmprj` which can be used to view the sample particle datasets.

Although, it is possible to host multiple instance descriptions in a single project file it is recommended to only have one description per file. Both files define a *view*, which is the only node in the top-level node `MegaMol`. The other keywords describe the behavior of this view.

>>>>>>> 0ae2f4429 (manual update ...)
Example 1: `simple_siff.mmprj`
```xml
    <?xml version="1.0" encoding="utf-8"?>
    <MegaMol type="project" version="1.3">
    <view name="dataview" viewmod="view">
        <module class="SIFFDataSource" name="data" />
        <module class="SphererRenderer" name="renderer" />
        <module class="View3D" name="view" />
        <module class="LinearTransferFunction" name="colors">
                <param name="mincolour" value="forestgreen" />
                <param name="maxcolour" value="lightskyblue" />
        </module>
        <module class="ClipPlane" name="clipplane">
                <param name="colour" value="#80808000" />
        </module>
        <module class="ScreenShooter" name="screenshooter">
                <param name="view" value="inst" />
        </module>
        <call class="MultiParticleDataCall" from="renderer::getdata" to="data::getdata" />
        <call class="CallRender3D" from="view::rendering" to="renderer::rendering" />
        <call class="CallGetTransferFunction" from="renderer::gettransferfunction" to="colors::gettransferfunction" />
        <call class="CallClipPlane" from="renderer::getclipplane" to="clipplane::getclipplane" />
    </view>
    </MegaMol>
```

Example 2: `pdbcartoonview.mmprj`
```xml
    <?xml version="1.0" encoding="utf-8"?>
    <MegaMol type="project" version="1.3">
    <view name="pdbcartoonview" viewmod="view3d">
        <module class="PDBLoader" name="::pdbdata" />
        <module class="MoleculeCartoonRenderer" name="cartoonren" />
        <module class="View3d" name="view3d" />
        <call class="CallRender3D" from="view3d::rendering" to="cartoonren::rendering" />
        <call class="MolecularDataCall" from="cartoonren::getdata" to="::pdbdata::dataout" />
    </view>
    </MegaMol>
>>>>>>> 800f17d5c (manual update)

*Parameter Slots* are the third type of slots. 
These are access points to exposed parameters controlling the functionality. 
Such parameters are automatically included in the frontend’s GUI. 
An example of such parameters is the setup of the virtual camera in modules of type `View3D_2` or the dataset file name in data source modules.  

The *module call graph* is configured for MegaMol using a project file. 
These files define modules and interconnecting calls for different instance specifications. 

<!-- DEPERCATED

<<<<<<< HEAD
There are two types of instances:
Views (see section [Views](#views)) and jobs (see section [Jobs](#jobs)). 
The starting command line of the console frontend loads project files and requests instantiation of views and jobs.
=======
```xml
        <module class="MoleculeCartoonRenderer" name="cartoonren" />
        <module class="SphererRenderer" name="renderer" />
```
>>>>>>> 800f17d5c (manual update)

<<<<<<< HEAD
-->
=======
After this, the *view* module is specified.
All used data source modules use mainly slots with the same names, i.e. a *ParameterSlot* named `filename` and a *CalleeSlot* named `getdata`, compatible with MultiParticleDataCall, providing access to the loaded data. Specifying the right config set variable thus allows the caller to use data sets from different file formats with this project file. See the online documentation for more information on these file formats. The recommended file format for MegaMol currently is **MMPLD**, and the corresponding data source module is thus the default module.
>>>>>>> 0ae2f4429 (manual update ...)

<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
#### Views 
=======
```xml
        <module class="View3d" name="view3d" />
        <module class="View3D" name="view" />
        <module class="LinearTransferFunction" name="colors">
                <param name="mincolour" value="forestgreen" />
                <param name="maxcolour" value="lightskyblue" />
        </module>
        <module class="ClipPlane" name="clipplane">
                <param name="colour" value="#80808000" />
        </module>
```
>>>>>>> 800f17d5c (manual update)

*Views* are one of the two instance types MegaMol can run. 
They are specified by the corresponding tag in a MegaMol project file (see section [Project Files](#project-files)). 
When a view is instantiated, a corresponding namespace will be created, and all modules instantiated as part of the view will be created inside this namespace. 

<<<<<<< HEAD
<!-- DEPRECATED
=======
The following block deals with the modules being interconnected using call objects. The corresponding tags specify the class of the call, the source *CallerSlot* to connect `from`, and the targeted *CalleeSlot* to connect `to`. The slot identifiers consist of module instance name (as defined in the project file, here its `cartoonren` and `::pdbdata`, while the slot name is as defined by the implementation (i.e. `rendering`, `getdata` and `dataout`). Specifying the full name would require the instance name this view will be instantiated as. Searching for the slots does, therefore, work using relative names.
>>>>>>> 0ae2f4429 (manual update ...)

<<<<<<< HEAD
For example, the project file seen in next section ([Project Files](#project-files)) defines the module data as part of the view dataview. 
If this view is instantiated by the command line note (note that files might need some path adjustments):
=======
```xml
        <call class="CallRender3D" from="view3d::rendering" to="cartoonren::rendering" />
        <call class="MolecularDataCall" from="cartoonren::getdata" to="::pdbdata::dataout" />
        <call class="MultiParticleDataCall" from="renderer::getdata" to="data::getdata" />
        <call class="CallRender3D" from="view::rendering" to="renderer::rendering" />
        <call class="CallGetTransferFunction" from="renderer::gettransferfunction" to="colors::gettransferfunction" />
        <call class="CallClipPlane" from="renderer::getclipplane" to="clipplane::getclipplane" />
```
>>>>>>> 800f17d5c (manual update)

    $ ./megamol.sh -p ../docs/samples/projects/pdbcartoonview.mmprj -i pdbcartoonview pv --paramfile ../docs/samples/projects/pdbmolview02.param -v ::pdbdata::pdbFilename ../docs/samples/sampledata/1m40_sim.pdb -v ::pdbdata::xtcFilename ../docs/samples/sampledata/1m40_100frames.xtc

<<<<<<< HEAD
Then the module will be created with the full name `::inst::data`. 
Correspondingly, its parameter slot `filename` can be globally addressed by `::inst::data::filename`. 
This allows for the instantiation of several independent view instances. 
For each view instance, a rendering window will be created. 
To provide the content for the rendering window, each view instance description needs to provide a *default view*, usually via the `viewmod` attribute of the view tag in the MegaMol project file. 
The value of this attribute is the name for the view module to be called by the window management code. 
This module class must be implemented by deriving from `::megamol::core::view::AbstractView`. 
Typically, you use `View3D` or `View2D`.
MegaMol provides some internal description of views which can be instantiated without loading a project file first. 
The view description *TestSpheres* used in section [Test](#test-installation) is one example of such a built-in description.
-->

<!-- ---------------------------------------------------------------------- -->
### View Interaction
=======
```xml
        <module class="ScreenShooter" name="screenshooter">
                <param name="view" value="inst" />
        </module>
```
>>>>>>> 800f17d5c (manual update)

<<<<<<< HEAD
The primary interaction with a view is controlling the camera with mouse and keyboard. 
The keyboard mapping is implemented by button parameters of the view module, also available in the GUI. 
Most parameters can be found in the sub-namespace `viewKey` inside the view name, e.g. `RotLeft`. 
The corresponding parameter button in the GUI also shows the associated hotkey.
=======
**Note**: If you experience problems with one of the renderers, for example, due to problems with your graphics card or graphics driver, try to select another one by specifying it in line 7, i.e., change the *class* value from `SphererRenderer` to `SimpleGeoSphereRenderer`.
>>>>>>> 0ae2f4429 (manual update ...)

<<<<<<< HEAD
***UPDATE REQUIRED***:  
<!-- TODO -->
=======
--> 

=======
>>>>>>> cdd0bbbc9 (docu)
<!-- ###################################################################### -->
-----
## MegaMol Graph

In this chapter, we show the operating principle of MegaMol which is required to creating own custom projects for MegaMol.

<!-- ---------------------------------------------------------------------- -->
### Modules, Views and Calls

The runtime functionality of MegaMol is constructed by *modules* and *calls*. 
These two type of objects are instantiated at runtime, interconnected and build the *module call graph*. 
The figure given below, shows an example module call graph containing a *view*, the rendering content of a window, a *renderer*, a *data source*, and some modules providing additional information for the renderer. 
The modules are interconnected by the *call* objects. 
The connection endpoints at the modules are *CallerSlots* (outgoing, located on the right side of modules) or *CalleeSlots* (incoming, located on the left side of modules) shown as circled dots.

The module call graph follows the pull pattern. 
This means that modules request function invocation by other modules. 
For example, the *view* module needs to update the window content. 
The *view* module thus invokes the *renderer* module to provide a new rendering. 
The *renderer* calls the data source if new data is available or to provide the old cached data.

Left-most module view of class `View3D_2` represents the rendering content of a window. 
The center module renderer of class `BoundingBoxRenderer` and `SphererRenderer` are called subsequently by the window using the corresponding call of type `CallRenderer3D_2`. 
The right modules provide data and additional information for the renderer, namely a color map transfer function and a clipping plane. 

![Example Graph](pics/example_graph.png)  

*Example module call graph.*

<!-- ---------------------------------------------------------------------- -->
#### Modules and Calls

*Modules* are the functional entities of MegaMol. 
They provide several programmatic access points, the *slots*. 
Two types of these slots are shown in the above figure of an example graph as circled dots.

*CalleeSlots* are access points of modules, through which these can be called to perform a function. 
For example, modules of class `SphererRenderer` provide a CalleeSlot `rendering` through which the rendering function can be invoked. 
The counterparts are CallerSlots which are outgoing access points. 
These allow modules to call other modules. 
View modules of class `View3D_2` provide a corresponding slot `rendering` to call a connected renderer.
These two types of slots are connected using objects of *call* classes. 
Both *CalleeSlots* and *CallerSlots* specify types of calls they are compatible with. 
In the case of the above examples of renderings-relates slots, this is the type `CallRender3D_2`.

*Calls* should be lightweight. 
They are thin interfaces meant for data transport. 
For example, data to be visualized is loaded by data source modules. 
In the example graph figure the module *data* of class *MMPLDDataSource* loads a specified data set into main memory and provides the data through its *CalleeSlot*. 
The data is accessed through a *MultiParticleDataCall*. 
The call, however, does not copy the data but provides access to the data in terms of memory pointers, and metadata. 
This avoidance of copy operations is most important and one of the core design ideas of MegaMol.

*Parameter Slots* are the third type of slots. 
These are access points to exposed parameters controlling the functionality. 
Such parameters are automatically included in the frontend’s GUI. 
An example of such parameters are the setup of the virtual camera in modules of type `View3D_2` or the dataset file name in data source modules.  

The *module call graph* is configured for MegaMol using a project file. 
These files define modules and interconnecting calls for different instance specifications. 

<!-- DEPERCATED

There are two types of instances:
Views (see section [Views](#views)) and jobs (see section [Jobs](#jobs)). 
The starting command line of the console frontend loads project files and requests instantiation of views and jobs.

-->

<!-- ---------------------------------------------------------------------- -->
#### Views 

*Views* are one of the two instance types MegaMol can run. 
They are specified by the corresponding tag in a MegaMol project file (see section [Project Files](#project-files)). 
When a view is instantiated, a corresponding namespace will be created, and all modules instantiated as part of the view will be created inside this namespace. 

<!-- DEPRECATED

For example, the project file seen in next section ([Project Files](#project-files)) defines the module data as part of the view dataview. 
If this view is instantiated by the command line note (note that files might need some path adjustments):

    $ ./megamol.sh -p ../docs/samples/projects/pdbcartoonview.mmprj -i pdbcartoonview pv --paramfile ../docs/samples/projects/pdbmolview02.param -v ::pdbdata::pdbFilename ../docs/samples/sampledata/1m40_sim.pdb -v ::pdbdata::xtcFilename ../docs/samples/sampledata/1m40_100frames.xtc

Then the module will be created with the full name `::inst::data`. 
Correspondingly, its parameter slot `filename` can be globally addressed by `::inst::data::filename`. 
This allows for the instantiation of several independent view instances. 
For each view instance, a rendering window will be created. 
To provide the content for the rendering window, each view instance description needs to provide a *default view*, usually via the `viewmod` attribute of the view tag in the MegaMol project file. 
The value of this attribute is the name for the view module to be called by the window management code. 
This module class must be implemented by deriving from `::megamol::core::view::AbstractView`. 
Typically, you use `View3D` or `View2D`.
MegaMol provides some internal description of views which can be instantiated without loading a project file first. 
The view description *TestSpheres* used in section [Test](#test-installation) is one example of such a built-in description.
-->

<!-- ---------------------------------------------------------------------- -->
### View Interaction
>>>>>>> 800f17d5c (manual update)

<<<<<<< HEAD
Some useful controls:
- Hitting *Home* (aka *Pos1*) is associated with the button *resetView*. This function resets the view to default.
<<<<<<< HEAD
=======
The primary interaction with a view is controlling the camera with mouse and keyboard. 
The keyboard mapping is implemented by button parameters of the view module, also available in the GUI. 
Most parameters can be found in the sub-namespace `viewKey` inside the view name, e.g. `RotLeft`. 
The corresponding parameter button in the GUI also shows the associated hotkey.

***UPDATE REQUIRED***:  
<!-- TODO -->

Some useful controls:
- Hitting *Home* (aka *Pos1*) is associated with the button *resetView*. This function resets the view to default.
>>>>>>> ee6adca4d (docu)
- Hold the *Left Mouse Button* and move your mouse to rotate the view around the look-at point. 
    The look-at point initially is placed in the center of the bounding box of the data set.
- Hold *Shift* while holding and dragging the *Left Mouse Button* rolls the camera around the viewing direction.
- Hold *Control* while holding and dragging the *Left Mouse Button* rotates the camera around its center point.
- Hold *Alt* while holding and dragging the *Left Mouse Button* moves the camera orthogonally to the viewing direction.
- Hold the *Middle Mouse Button* and move your mouse up or down to zoom the view by move the camera forwards or backwards. 
    Note that if you zoom in too much, parts of the data set will be clipped by the near-clipping plane.
- Hold *Alt* while holding and dragging the *Middle Mouse Button* zoom the view by changing the opening angle of the camera.
- Hold *Control* while holding and dragging the *Middle Mouse Button* moves the look-at point forwards or backwards, changing the center of the corresponding rotation. 
    Use the parameter `showLookAt` of the view to visualize the look-at point for better adjustment.
<<<<<<< HEAD

<<<<<<< HEAD

<!-- ###################################################################### -->
-----
## Project Files

Project files are [`lua`](https://www.lua.org/) scripts using special custom functions to define any module graph for MegaMol.
Some predefined example project script files are available in the `examples` directory, which is installed next to the `bin` directory.
Here you can see the example project script `..\examples\testspheres_megamol.lua`:

<<<<<<< HEAD
```lua
    mmCreateView("testspheres", "View3D_2","::view")
=======
<!-- ---------------------------------------------------------------------- -->
<a name="makescreenshot"></a>
>>>>>>> 800f17d5c (manual update)

    mmCreateModule("BoundingBoxRenderer","::bbox")
    mmCreateModule("DistantLight","::distantlight")
    mmCreateModule("SphereRenderer","::renderer")
    mmCreateModule("TestSpheresDataSource", "::data")

<<<<<<< HEAD
    mmSetParamValue("::renderer::renderMode", [=[Simple]=])
=======
<!-- ADD GUI menu option: `Screenshot`-->
=======
[//]: # (----------------------------------------------------------------------) 
### Making High-Resolution Screenshots

[//]: # (ADD GUI menu option: `Screenshot`-->
>>>>>>> 4fa438626 (manual update ...)

MegaMol has special functions to create high-resolution screen shoots of any rendering, namely the `ScreenShooter` module. The provided starting scripts add this module. If you create a project file of your own, remember to add the `ScreenShooter` module. The corresponding settings can be found in the AntTweakBar in the groups `inst::screenshooter` and `inst::screenshooter::anim` (see figure [ScreenShooter](#screenshooter)).
>>>>>>> 800f17d5c (manual update)
=======
- Hold the *left mouse button* and move your mouse to rotate the view around the look-at point. The look-at point initially is placed in the center of the bounding box of the data set.
- Hold *shift* while holding and dragging the *left mouse button* rolls the camera around the viewing direction.
- Hold *control* while holding and dragging the *left mouse button* rotates the camera around its center point.
- Hold *alt* while holding and dragging the *left mouse button* moves the camera orthogonally to the viewing direction.
- Hold the *middle mouse button* and move your mouse up or down to zoom the view by move the camera forwards or backwards. Note that if you zoom in too much, parts of the data set will be clipped by the near-clipping plane.
- Hold *alt* while holding and dragging the *middle mouse button* zoom the view by changing the opening angle of the camera.
- Hold *control* while holding and dragging the *middle mouse button* moves the look-at point forwards or backwards, changing the center of the corresponding rotation. Use the parameter
`showLookAt` of the view to visualize the look-at point for better adjustment.
=======
>>>>>>> ee6adca4d (docu)


<!-- ###################################################################### -->
-----
## Project Files

Project files are [`lua`](https://www.lua.org/) scripts using special custom functions to define any module graph for MegaMol.
Some predefined example project script files are available in the `examples` directory, which is installed next to the `bin` directory.
Here you can see the example project script `..\examples\testspheres_megamol.lua`:

```lua
    mmCreateView("testspheres", "View3D_2","::view")

    mmCreateModule("BoundingBoxRenderer","::bbox")
    mmCreateModule("DistantLight","::distantlight")
    mmCreateModule("SphereRenderer","::renderer")
    mmCreateModule("TestSpheresDataSource", "::data")

    mmSetParamValue("::renderer::renderMode", [=[Simple]=])

    mmCreateCall("CallRender3D_2", "::view::rendering", "::bbox::rendering")
    mmCreateCall("CallRender3D_2","::bbox::chainRendering","::renderer::rendering")
    mmCreateCall("MultiParticleDataCall", "::renderer::getData", "::data::getData")
    mmCreateCall("CallLight","::renderer::lights","::distantlight::deployLightSlot")
```

Project files can easily be created using the built in *Configurator*.
It can be opened via the menu `Windows / Configurator`.
You can either edit the currently running MegaMol graph (which might be empty) or you can create a new project starting a module graph by adding the main view module `View3D_2`.
A detailed description of the configurator can be found in the readme file of the [GUI plugin](../plugins/gui#configurator).

<!-- TODO 

Add more ... ?

-->

<!-- ###################################################################### -->
-----
## Making High-Resolution Screenshots

The GUI menu option `Screenshot` (hotkey `F2`) provides a basic screenshot funtionality using the current viewport size. 
If screenshots are taken consecutively, the given file name is prepended by an incrementing suffix. 
This way, no new file name has to be set after each screenshot.  

<<<<<<< HEAD
For a more flexible way, use Screenshoter module
<!-- DEPRECATED 
MegaMol has special functions to create high-resolution screen shoots of any rendering, namely the `ScreenShooter` module. The provided starting scripts add this module. If you create a project file of your own, remember to add the `ScreenShooter` module. The corresponding settings can be found in the ImGui in the groups `inst::screenshooter` and `inst::screenshooter::anim` (see figure [ScreenShooter](#screenshooter)).
>>>>>>> 0ae2f4429 (manual update ...)

    mmCreateCall("CallRender3D_2", "::view::rendering", "::bbox::rendering")
    mmCreateCall("CallRender3D_2","::bbox::chainRendering","::renderer::rendering")
    mmCreateCall("MultiParticleDataCall", "::renderer::getData", "::data::getData")
    mmCreateCall("CallLight","::renderer::lights","::distantlight::deployLightSlot")
```

<<<<<<< HEAD
Project files can easily be created using the built in *Configurator*.
It can be opened via the menu `Windows / Configurator`.
You can either edit the currently running MegaMol graph (which might be empty) or you can create a new project starting a module graph by adding the main view module `View3D_2`.
A detailed description of the configurator can be found in the readme file of the [GUI plugin](../plugins/gui#configurator).

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<!-- TODO 
=======
<!-- DEPERCATED 
=======
[//]: # (DEPERCATED 
>>>>>>> 4fa438626 (manual update ...)
![ScreenShooter](pics/screenshooter.png)
The parameter filename specifies the path to the image file to be created. MegaMol only creates PNG files. Hit the button trigger to have MegaMol create the requested screenshot.
-->
>>>>>>> f5eec258f (manual update ....)

<<<<<<< HEAD
<<<<<<< HEAD
Add more ... ?
=======
<!-- ---------------------------------------------------------------------- -->
=======
[//]: # (----------------------------------------------------------------------) 
>>>>>>> 4fa438626 (manual update ...)
=======
=======
-->

>>>>>>> 8b2342edf (docu)
<!-- DEPERCATED 
=======
<!-- DEPRECATED/UPDATE - MegaMol is not accessible for modules and therefore the required view instance can not be found

Offering more flexible options and special functions to create high-resolution screenshoots of any rendering, you can add the `ScreenShooter` module to you project.
The corresponding settings can be found in the modules parameters provided in the GUI (see figure of `ScreenShooter` parameters below).

In order to connect the `ScreenShooter` with your *view*, you need to set the **instance name** of your view instance in the corresponding variable `::screenshooter::view` (e.g. to `inst`). When making single screenshots, set the option `makeAnim` in the group `::screenshooter::anim` to `disabled`, as shown in the figure. 
Ignore the remaining options in that group. 
These options will be explained in section [Making Simple Videos](#making-simple-videos), as they are used to produce videos.

The parameters `imgWidth` and `imgHeight` specify the size of the screenshot to be rendered. 
These values are not limited to the window size and can be, in theory, arbitrarily large. 
If these values are getting large, the image can be rendered in several tiles, i.e., sub-images. 
The size for these tiles is specified by `tileWidth` and `tileHeight`. 
However, many renderers have problems with producing these tiled images. 
It is, thus, recommended to set `tileWidth` and `tileHeight` to be at least as large as `imgWidth` and `imgHeight`. 
The values for `tileWidth` and `tileHeight` are limited by the maximum texture size, maximum frame buffer object size and graphics memory size of your graphics card. 
Thus, these values are often limited.
The parameter file name specifies the path to the image file to be created. 
MegaMol only creates PNG files. 
Hit the button trigger to have MegaMol create the requested screenshot.

>>>>>>> ee6adca4d (docu)
![ScreenShooter](pics/screenshooter.png)

-->

<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
<<<<<<< HEAD
>>>>>>> 0ae2f4429 (manual update ...)
### Reproducibility
>>>>>>> 800f17d5c (manual update)

<<<<<<< HEAD
<<<<<<< HEAD
-->

<!-- ###################################################################### -->
-----
## Making High-Resolution Screenshots

The GUI menu option `Screenshot` (hotkey `F2`) provides a basic screenshot functionality using the current viewport size. 
If screenshots are taken consecutively, the given file name is prepended by an incrementing suffix. 
This way, no new file name has to be set after each screenshot.  

<!-- DEPRECATED/UPDATE - MegaMol is not accessible for modules and therefore the required view instance can not be found
=======
MegaMol stores the active project and all parameter settings in the EXIF field of the saved screenshots. Please note that this field currently contains a simple zero-terminated string with the LUA code required to reproduce the state when the screenshot is taken, and **not** valid EXIF data. Such a project can be restored by just loading the PNG file:
=======
MegaMol stores the active project and all parameter settings in the EXIF field of the saved screenshots. 
Please note that this field currently contains a simple zero-terminated string with the LUA code required to reproduce the state when the screenshot is taken, and **not** valid EXIF data. Such a project can be restored by simply loading the PNG file:
>>>>>>> 8b2342edf (docu)

    $ megamol.exe <something>.png

<<<<<<< HEAD
Also note that only Views with direct access to camera parameters (like View3D_2, but unlike the original View3D, which requires explicit serialization of camera parameters) can be properly restored.
>>>>>>> f5eec258f (manual update ....)
=======
Also note that only Views with direct access to camera parameters (like `View3D_2`) can be properly restored.
>>>>>>> 8b2342edf (docu)

<<<<<<< HEAD
<<<<<<< HEAD
Offering more flexible options and special functions to create high-resolution screenshots of any rendering, you can add the `ScreenShooter` module to you project.
The corresponding settings can be found in the modules parameters provided in the GUI (see figure of `ScreenShooter` parameters below).
=======
<!-- ---------------------------------------------------------------------- -->
<a name="makevideo"></a>

=======
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
### Making Simple Videos

<<<<<<< HEAD
<<<<<<< HEAD
<!-- ADD Cinematic plugin: [cinematic plugin](https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/cinematic/README.md) -->
>>>>>>> 800f17d5c (manual update)

In order to connect the `ScreenShooter` with your *view*, you need to set the **instance name** of your view instance in the corresponding variable `::screenshooter::view` (e.g. to `inst`). When making single screenshots, set the option `makeAnim` in the group `::screenshooter::anim` to `disabled`, as shown in the figure. 
Ignore the remaining options in that group. 
These options will be explained in section [Making Simple Videos](#making-simple-videos), as they are used to produce videos.

The parameters `imgWidth` and `imgHeight` specify the size of the screenshot to be rendered. 
These values are not limited to the window size and can be, in theory, arbitrarily large. 
If these values are getting large, the image can be rendered in several tiles, i.e., sub-images. 
The size for these tiles is specified by `tileWidth` and `tileHeight`. 
However, many renderers have problems with producing these tiled images. 
It is, thus, recommended to set `tileWidth` and `tileHeight` to be at least as large as `imgWidth` and `imgHeight`. 
The values for `tileWidth` and `tileHeight` are limited by the maximum texture size, maximum frame buffer object size and graphics memory size of your graphics card. 
Thus, these values are often limited.
The parameter file name specifies the path to the image file to be created. 
MegaMol only creates PNG files. 
Hit the button trigger to have MegaMol create the requested screenshot.
=======
[//]: # (----------------------------------------------------------------------) 
### Making Simple Videos

[//]: # (ADD Cinematic plugin: [cinematic plugin](https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/cinematic/README.md)) 
>>>>>>> 4fa438626 (manual update ...)
=======
<!-- ---------------------------------------------------------------------- -->
### Making Simple Videos
=======

<!-- ###################################################################### -->
-----
## Making Simple Videos
>>>>>>> cdd0bbbc9 (docu)

<<<<<<< HEAD
<!-- ADD Cinematic plugin: [cinematic plugin](https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/cinematic/README.md) -->
>>>>>>> 0ae2f4429 (manual update ...)

![ScreenShooter](pics/screenshooter.png)

-->


<!-- ###################################################################### -->
-----
## Making Simple Videos

MegaMol cannot create video files directly. 
However, MegaMol can create a sequence of screenshots of a time-dependent data set showing the different points-in-time. 
This functionality is provided and further described in the [*Cinematic* plugin](../plugins/cinematic#usage).

<!-- DEPRECATED 

Adjust the parameters in the group `::screenshooter::anim` in addition to the parameters for simple screenshots ([ScreenShots](#makescreenshot)).
=======
*TODO*

See [Cinematic Plugin](../plugins/cinematic#usage).

<!-- DEPRECATED 
MegaMol cannot create video files directly. However, MegaMol can create a sequence of screenshots of a time-dependent data set showing the different points-in-time. Adjust the parameters in the group `inst::screenshooter::anim` in addition to the parameters for simple screenshots ([ScreenShots](#makescreenshot)).
>>>>>>> 8b2342edf (docu)
=======
MegaMol cannot create video files directly. 
However, MegaMol can create a sequence of screenshots of a time-dependent data set showing the different points-in-time. 
This functionality is provided and further described in the [*Cinematic* plugin](../plugins/cinematic#usage).

<!-- DEPRECATED 

Adjust the parameters in the group `::screenshooter::anim` in addition to the parameters for simple screenshots ([ScreenShots](#makescreenshot)).
>>>>>>> ee6adca4d (docu)
Enable the option `makeAnim` to make a screenshot sequence.

The parameters `from` and `to` specify the time codes to start and end the animation. 
The parameter `step` specifies the time code increment between two screenshots.
For example, if you specify `from=0`, `to=100`, and `step= 10` (assuming the dataset stores enough time frames), 11 screenshots will be created and stored. 
These will show the data set at times 0, 10, 20, ... , 90, and 100. 
The screenshot files names will be extended by an increasing number: e.g. `test.00000.png`, `test.00001.png`, `test.00002.png`, ...
This sequence of image files can then be merged to produce a video file, e.g. using avconv:

    $ avconv -r 30 -i test.%05d.png test.mp4

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> ee6adca4d (docu)
**KNOWN BUG**: 
Several renderers will request the best data they can get. 
As usually data is loaded asynchronously, the correct data is often not available yet, and the best data is the data from a slightly wrong time. 
While this is not a big deal for viewing data, it is fatal when rendering images for videos. 
Many renderers thus expose a parameter `forceTime`, or with a similar name. 
Set this parameter to `true` and the renderer will always show the correct data. 
It will need to wait if the correct data is not available, yet, which can reduce the overall performance.

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
-->
=======
=======

<<<<<<< HEAD
>>>>>>> f5eec258f (manual update ....)
<!-- ###################################################################### -->
<a name="jobs"></a>
>>>>>>> 800f17d5c (manual update)

<!-- ###################################################################### -->
<!-- DEPRECATED/UPDATE
=======
[//]: # (######################################################################) 
>>>>>>> 4fa438626 (manual update ...)
## Jobs

This chapter discusses the job concept available in MegaMol. 
Especially, how jobs can be used for data conversion. 
Examples are based on the project script files available in the installed `examples` directory.

<<<<<<< HEAD
<<<<<<< HEAD
=======
<!-- ---------------------------------------------------------------------- -->
=======
[//]: # (----------------------------------------------------------------------) 
>>>>>>> 4fa438626 (manual update ...)
=======
**KNOWN BUG**: several renderers will request the best data they can get. As usually data is loaded asynchronously, the correct data is often not available yet, and the best data is the data from a slightly wrong time. While this is not a big deal for viewing data, it is fatal when rendering images for videos. Many renderers thus expose a parameter `forceTime`, or with a similar name. Set this parameter to `true` and the renderer will always show the correct data. It will need to wait if the correct data is not available, yet, which can reduce the overall performance.

=======
**KNOWN BUG**: Several renderers will request the best data they can get. As usually data is loaded asynchronously, the correct data is often not available yet, and the best data is the data from a slightly wrong time. While this is not a big deal for viewing data, it is fatal when rendering images for videos. Many renderers thus expose a parameter `forceTime`, or with a similar name. Set this parameter to `true` and the renderer will always show the correct data. It will need to wait if the correct data is not available, yet, which can reduce the overall performance.
=======
>>>>>>> ee6adca4d (docu)
-->
>>>>>>> 8b2342edf (docu)

<!-- ###################################################################### -->
<!-- DEPRECATED/UPDATE
## Jobs

This chapter discusses the job concept available in MegaMol. 
Especially, how jobs can be used for data conversion. 
Examples are based on the project script files available in the installed `examples` directory.


<<<<<<< HEAD
<!-- ---------------------------------------------------------------------- -->
<<<<<<< HEAD
>>>>>>> 0ae2f4429 (manual update ...)
### Job instance
>>>>>>> 800f17d5c (manual update)

<<<<<<< HEAD
=======
>>>>>>> 6668c26ff (docu toc)
=======
>>>>>>> ee6adca4d (docu)
### Job Instance

Jobs are the second type of instances available at the MegaMol runtime (compare view instances in section [Views](#views)). 
The primary difference is the `<job>` tag as primary instance tag. 
Similarly, to the viewmod attribute, the `<job>` tag specifies a jobmod module as entry module.
<<<<<<< HEAD
=======
Jobs are the second type of instances available at the MegaMol runtime (compare view instances in section [Views](#views)). The primary difference is the `<job>` tag as primary instance tag. Similarly, to the viewmod attribute, the `<job>` tag specifies a jobmod module as entry module.
>>>>>>> 0ae2f4429 (manual update ...)
=======
>>>>>>> ee6adca4d (docu)

```xml
    <job name="convjob" jobmod="job">
```

One significant limitation of this release is that the MegaMol Configurator is only able to edit view instance descriptions, see section [Configurator](#configurator). 
If you want graphical assistance in creating a job description, the recommended way is to create a view instance description with all required modules and calls. 
Use a `DataWriterJob` module as an entry point. 
Save the corresponding project file and edit it manually with a text editor. 
Replace the `<view>` tags with the similarly behaving `<job>` tags and adjust the corresponding attributes.
<<<<<<< HEAD

=======
>>>>>>> ee6adca4d (docu)


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
The MegaMol Particle List Data file format (MMPLD) is a very fast loading binary memory dump of MegaMol meant for small and mid-sized datasets (1-10 mio. particles). MegaMol can convert most of its supported file formats to MMPLD file format. More precisely, all file formats which are loaded by a MegaMol module supporting the `MultiParticleDataCall` can be converted to MMPLD. For this, specify a converter job using a `DataWriterJob` module and a `MMPLDWriter` module. The content of the project file `makemmpld.mmprj` which can be used to convert data into the MMPLD file format, is shown below.
=======
<!-- UPDATE
=======
[//]: # (UPDATE
>>>>>>> 4fa438626 (manual update ...)
The MegaMol Particle List Data file format (MMPLD) is a very fast loading binary memory dump of MegaMol meant for small and mid-sized datasets (1-10 mio. particles). MegaMol can convert most of it’s supported file formats to MMPLD file format. More precisely, all file formats which are loaded by a MegaMol module supporting the `MultiParticleDataCall` can be converted to MMPLD. For this, specify a converter job using a `DataWriterJob` module and a `MMPLDWriter` module. The content of the project file `makemmpld.mmprj` which can be used to convert data into the MMPLD file format, is shown below.
>>>>>>> 800f17d5c (manual update)
=======
=======
*UPDATE*

>>>>>>> 8b2342edf (docu)
<!-- UPDATE
=======
### Converting to MMPLD

>>>>>>> ee6adca4d (docu)
The MegaMol Particle List Data file format (MMPLD) is a very fast loading binary memory dump of MegaMol meant for small and mid-sized datasets (1-10 mio. particles). MegaMol can convert most of its supported file formats to MMPLD file format. More precisely, all file formats which are loaded by a MegaMol module supporting the `MultiParticleDataCall` can be converted to MMPLD. For this, specify a converter job using a `DataWriterJob` module and a `MMPLDWriter` module. The content of the project file `makemmpld.mmprj` which can be used to convert data into the MMPLD file format, is shown below.
>>>>>>> 0ae2f4429 (manual update ...)

```xml
    <?xml version="1.0" encoding="utf-8"?>
    <MegaMol type="project" version="1.3">

    <job name="convjob" jobmod="job">
        <module class="SIFFDataSource" name="data" />

        <module class="DataWriterJob" name="job" />
        <module class="MMPLDWriter" name="writer" />
        <call class="DataWriterCtrlCall" from="job::writer" to="writer::control" />
        <call class="MultiParticleDataCall" from="writer::data" to="data::getdata" />

            </job>
    </MegaMol>
```

The entry module for data conversion is of class `DataWriterJob`, see line 11. This job module controls writing several files into a new data set. The output is implemented in corresponding write modules, like the `MMPLDWriter`, see line 12. This writer module is then connected to a module providing the data. In the simplest scenario, this is directly a data loader module. The above example selects one module from several options, in the same way the data viewing project does (see section [Project Files](#project-files)). The job is instantiated similarly using the command line. The paths for `makemmpld.mmprj` and `inputfile.siff` might need to be adjusted:

    $ mmconsole -p makemmpld.mmprj -i convjob j -v j::data::filename inputfile.siff -v j::writer::filename outputfile.mmpld

The input file name and output file name are explicitly specified using the -v arguments. 
The job execution starts immediately. 
After data is written, MegaMol terminates itself. 

To convert from other file formats, for which a corresponding loader does exist, you should be able to adjust this project file.

--> 

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
--> 
=======
<!-- ###################################################################### -->
<<<<<<< HEAD
<a name="advanced-usage"></a>

=======
[//]: # (######################################################################) 
>>>>>>> 4fa438626 (manual update ...)
=======
=======
<!-- ---------------------------------------------------------------------- -->
=======
<!-- ###################################################################### -->
-----
>>>>>>> 83aa4eadb (docu)
## Reproducibility

MegaMol stores the active project and all parameter settings in the EXIF field of the saved screenshots. 
Please note that this field currently contains a simple zero-terminated string with the LUA code required to reproduce the state when the screenshot is taken, and **not** valid EXIF data. Such a project can be restored by simply loading the PNG file:

    $ megamol.exe <something>.png

Also note that only Views with direct access to camera parameters (like `View3D_2`) can be properly restored.
<<<<<<< HEAD


>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
<!-- ###################################################################### -->
>>>>>>> 0ae2f4429 (manual update ...)
=======
<!-- MOVE to separate file 

>>>>>>> 8b2342edf (docu)
## Advanced Usage

This chapter discusses advanced usage of MegaMol.

-- >
<!-- DEPRECATED 

### Configurator (Windows)

The Configurator is a utility application for editing MegaMol project files. More specifically, it allows us to edit the modules, calls, and parameters to be instantiated and set for view instances (see sections [Modules, View, and Calls](#modules-views-calls) and [Project Files](#project-files)). The [image](#configurator-pic) below shows the main GUI window of the application.

<center>
<a name="configurator-pic"></a>
<img src="pics/configurator.png" alt="Configurator" style="width: 1024px;"/>
<p style="width: 1024px;">
The Configurator main GUI.
</p>
</center>

The Configurator is written in C#. The Configurator can be run on Linux using Mono 1. However, we never tested the Configurator with Mono and thus will not support it. The right editing pane allows you to place modules and interconnect their *CallerSlots* and *CalleeSlots* with matching calls. The tabs above that pane allow to open and edit several project files at once. On the left side, you have the list of modules available and, below, the parameters of the currently selected module (which is marked in the editing pane by a dashed border). The menu strip contains direct access to all functions. The Configurator is a stand-alone editor for MegaMol project files. There is no way to connect the editor to a running instance of MegaMol , e.g., to edit parameters on the fly.

### Class Data Base

To fill the left list of available modules, the Configurator requires a database containing all classes of the MegaMol module graph engine. This file is called the state file. You can load and save the state file using the Settings menu. There, you can also view a summary of the contents of the loaded state file. The MegaMol release package contains a state file of all modules and classes available in the released MegaMol binary. You can directly load this state file.

On Windows operating systems you can also analyze MegaMol binaries to generate a state file containing their classes. You find this function also in the Settings menu. This function relies on system DLLs and native, unmanaged code. So, this function will not work on Linux with Mono. A dialog window will open in which you can select the folder to analyze. After a short time, the list in this dialog will show you all compatible MegaMol binaries, including versions of the core library and all plugins. Select all binaries you want to be included in the state file. Be aware that you can only select binaries which depend on the exact same core library. Select Ok, and
the new state file will be created and loaded. Per default, a dialog will appear, allowing you to save the newly generated state file.

### Editing

Project files can be saved and loaded using the corresponding functions for the menu bar. Save will always ask you for a file name, but will use the last file name as the default value. New creates a new editing tab. To close a tab use the x button on the right or click a tab with the middle mouse button.

#### Modules

To add a new module to the current project, double-click on the corresponding entry in the module list. The module will be created in the upper left corner and will be automatically selected. To select a module click it with any mouse button. You can move all modules by simple dragging and dropping them with the mouse. This has no influence on the functionality. To delete the selected module hit the `Del` key.
While a module is selected, the parameter editor on the left shows all parameters of that module. In addition to the parameters exposed by *parameter slots*, there are some management parameters. You can edit the name of all module instances. For view modules, you can select if this module is the main view module for the whole project.
Click on a free space to deselect any module, call or slot. This will select the project itself.
You can now edit project parameters in the parameter editor on the left.

#### Calls

To select a *CallerSlot* (red triangles) or a *CalleeSlot* (green triangles), click the corresponding triangle of the module. The selected slot will be highlighted in yellow. All matching slots that are compatible with the selected slot are highlighted by a yellow outline. If the *Filter Compatible* option is active, the module list will then only show modules which have compatible slots. To connect two slots, drag with your mouse a line from one onto the other. The drag direction does not matter. If both slots are compatible, a matching call object is created and shown by a gray box. These call boxes are placed automatically. If the connection could be established by different call classes, a dialog window will appear asking to you select the call class to be used. To delete a call, select it by clicking on its box and press the `Del` key.
A (red) *CallerSlot* can only have a single outgoing call, while (green) *CalleeSlots* may be connected by multiple calls. Therefore, if you drag from a *CallerSlot* to a *CalleeSlot* which already has incoming connections, a new connection is added. If you, however, drag from a *CalleeSlot to a *CallerSlot* which already has an outgoing connection, this connection is replaced.

### Starting MegaMol

<<<<<<< HEAD
The Configurator allows you to directly start MegaMol with the currently selected project file. Remember: the Configurator is only an editor for project files and has no online connection to a running MegaMol instance. You thus need to remember to save your edited project files before starting MegaMol .
>>>>>>> 800f17d5c (manual update)
=======
The Configurator allows you to directly start MegaMol with the currently selected project file. Remember: The Configurator is only an editor for project files and has no online connection to a running MegaMol instance. You thus need to remember to save your edited project files before starting MegaMol.
>>>>>>> 0ae2f4429 (manual update ...)

<!-- ###################################################################### -->
-----
## Reproducibility

<<<<<<< HEAD
MegaMol stores the active project and all parameter settings in the EXIF field of the saved screenshots. 
Please note that this field currently contains a simple zero-terminated string with the LUA code required to reproduce the state when the screenshot is taken, and **not** valid EXIF data. Such a project can be restored by simply loading the PNG file:
=======
After launching the Configurator, selecting the menu item `MegaMol Start Arguments...` opens the dialog window shown in the above [figure](#configurator-start). Here you can edit all start settings. The menu item `Start MegaMol` directly starts MegaMol with these settings. So does the `Start` button in this dialog window. The `Ok` button stores changes to the settings and closes the dialog window without starting MegaMol. The top line of elements controls the way MegaMol starts. Tick directly to directly spawn the MegaMol process from the configurator. This is the only working option for Linux operating systems. Tick `in Cmd` or `in Windows Powershell` and the configurator will first start the corresponding shell and then spawn the MegaMol process inside this shell. Tick the `Keep Open` option to start these shells in a way that they will remain open after the MegaMol process terminates. 
>>>>>>> 0ae2f4429 (manual update ...)

    $ megamol.exe <something>.png

<<<<<<< HEAD
Also note that only Views with direct access to camera parameters (like `View3D_2`) can be properly restored.
=======
The whole process command line can be copied to the clipboard using the remaining menu items in the Start menu. The differences for Cmd shell and Powershell are only the way how special characters and strings are escaped.

-->
<<<<<<< HEAD

<!-- ADD:
    ## Add own plugin using the template
    1. Copy the template folder
    2. Rename the copied folder to the intended plugin name
    3. Execute the instawiz.pl script inside the new folder
        1. The script detects the plugin name
        2. Autogenerate the GUID
    4. Remove instawiz.pl
    5. Add libraries/dependencies to `CMakeLists.txt` (optional)
    6. Implement the content of your plugin
    7. Write a `Readme.md` for your plugin (mandatory)
    8. Add the folder to your local git
-->
>>>>>>> 800f17d5c (manual update)
=======
>>>>>>> 1a324379f (moved mmpld specs to separate folder, deguide ...)
=======
>>>>>>> ee6adca4d (docu)
