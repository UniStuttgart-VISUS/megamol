# MegaMol
MegaMol is a visualization middleware used to visualize point-based molecular data sets. This software is developed within the ​Collaborative Research Center 716, subproject ​D.3 at the ​[Visualization Research Center (VISUS)](https://www.visus.uni-stuttgart.de/institut.html) of the University of Stuttgart and at the ​Computer Graphics and Visualization Group of the TU Dresden.  

MegaMol succeeds [​MolCloud](http://www.visus.uni-stuttgart.de/institut/personen/wissenschaftliche-mitarbeiter/sebastian-grottel/molcloud.html), which has been developed at the University of Stuttgart in order to visualize point-based data sets. MegaMol™ is written in C++, and uses an OpenGL as Rendering-API and GLSL-Shader. It supports the operating systems Microsoft Windows and Linux, each in 32-bit and 64-bit versions. In large parts, MegaMol™ is based on [VISlib](https://svn.vis.uni-stuttgart.de/trac/vislib), a C++-class library for scientific visualization, which has also been developed at the University of Stuttgart. 

## Building MegaMol
### Linux
1. Clone the MegaMol repository
2. Create a build folder
3. Invoke `cmake` inside the build folder
4. Use `make` to build MegaMol
5. Use `make install` to create your MegaMol installation
6. Test Megamol with

        ./megamol.sh -i testspheres inst

### Windows

1. Use the cmake GUI to configure MegaMol
    1. The configuration creates a `sln` file inside the build folder
2. Open the `sln` file with *Visual Studio*
3. Use the `ALL_BUILD` target to build MegaMol
4. Use the `INSTALL` target to create your MegaMol installation
6. Test Megamol with

        console.exe -i testspheres inst

## Using the plugin template
1. Copy the template folder
2. Rename the copied folder to the intended plugin name
3. Execute the instawiz.pl script inside the new folder
    1. The script detects the plugin name
    2. Autogenerate the GUID
4. Remove instawiz.pl and Readme.md
5. Add the folder to your local git
6. Add libraries/dependencies to `CMakeLists.txt`
7. Implement the content of your plugin