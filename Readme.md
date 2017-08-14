
![](logo.png)  
[![Build Status Travis](https://travis-ci.org/UniStuttgart-VISUS/megamol.svg?branch=master)](https://travis-ci.org/UniStuttgart-VISUS/megamol)
[![Build Status AppVeyor](https://ci.appveyor.com/api/projects/status/github/UniStuttgart-VISUS/megamol?branch=master&svg=true)](https://ci.appveyor.com/api/projects/status/github/UniStuttgart-VISUS/megamol)  
MegaMol is a visualization middleware used to visualize point-based molecular data sets. This software is developed within the ​Collaborative Research Center 716, subproject ​D.3 at the ​[Visualization Research Center (VISUS)](https://www.visus.uni-stuttgart.de/institut.html) of the University of Stuttgart and at the ​Computer Graphics and Visualization Group of the TU Dresden.  

MegaMol succeeds [​MolCloud](http://www.visus.uni-stuttgart.de/institut/personen/wissenschaftliche-mitarbeiter/sebastian-grottel/molcloud.html), which has been developed at the University of Stuttgart in order to visualize point-based data sets. MegaMol is written in C++, and uses an OpenGL as Rendering-API and GLSL-Shader. It supports the operating systems Microsoft Windows and Linux, each in 32-bit and 64-bit versions. In large parts, MegaMol is based on [VISlib](https://svn.vis.uni-stuttgart.de/trac/vislib), a C++-class library for scientific visualization, which has also been developed at the University of Stuttgart. 

## Building MegaMol
### Linux
1. Clone the MegaMol repository
2. Create a build folder
3. Invoke `cmake` inside the build folder
4. Execute `make` to build MegaMol
5. Run `make install` to create your MegaMol installation
6. Test Megamol with

        ./megamol.sh -i testspheres inst

### Windows

1. Clone the MegaMol repository
2. Use the cmake GUI to configure MegaMol
    1. The configuration creates a `sln` file inside the build folder
3. Open the `sln` file with *Visual Studio*
4. Use the `ALL_BUILD` target to build MegaMol
5. Use the `INSTALL` target to create your MegaMol installation
6. Test Megamol with

        mmconsole.exe -i testspheres inst


## MegaMol Configurator
MegaMol offers a configurator GUI (C#) that runs with .Net Framework 4.
It runs also on Linux with Mono 3.2.8 (except for the analysis function and indirect-start functions).  
Download link: [MegaMolConfigurator](https://svn.vis.uni-stuttgart.de/trac/megamol/raw-attachment/wiki/WikiStart/MegaMolConfigurator.zip)


## How to use MegaMol
A detailed description of the MegaMol user interface see chapters `3-5` and appendix `A` in the 
[Handbook](https://svn.vis.uni-stuttgart.de/trac/megamol/raw-attachment/wiki/WikiStart/MegaMol_v1_1_Manual.pdf).

## Using the plugin template
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


## Citing MegaMol
Please use one of the following methods to reference the MegaMol project.


**MegaMol – A Prototyping Framework for Particle-based Visualization**  
S. Grottel, M. Krone, C. Müller, G. Reina, T. Ertl  
Visualization and Computer Graphics, IEEE Transactions on, vol.21, no.2, pp. 201--214, Feb. 2015  
doi: 10.1109/TVCG.2014.2350479

    @article{grottel2014megamol,
      author={Grottel, S. and Krone, M. and Muller, C. and Reina, G. and Ertl, T.},
      journal={Visualization and Computer Graphics, IEEE Transactions on},
      title={MegaMol -- A Prototyping Framework for Particle-based Visualization},
      year={2015},
      month={Feb},
      volume={21},
      number={2},
      pages={201--214},
      keywords={Data models;Data visualization;Graphics processing units;Libraries;Rendering(computer graphics);Visualization},
      doi={10.1109/TVCG.2014.2350479},
      ISSN={1077-2626}
    }
#
**Coherent Culling and Shading for Large Molecular Dynamics Visualization**  
S. Grottel, G. Reina, C. Dachsbacher, T. Ertl  
Computer Graphics Forum (Proceedings of EUROVIS 2010), 29(3):953 - 962, 2010

    @article{eurovis10-grottel,
      author = {Grottel, S. and Reina, G. and Dachsbacher, C. and Ertl, T.},
      title  = {{Coherent Culling and Shading for Large Molecular Dynamics Visualization}},
      url    = {http://go.visus.uni-stuttgart.de/megamol},
      year   = {2010},
      pages  = {953--962},
      journal = {{Computer Graphics Forum}},
      volume = {{29}},
      number = {{3}}
    }
#
**Optimized Data Transfer for Time-dependent, GPU-based Glyphs**  
S. Grottel, G. Reina, T. Ertl  
In Proceedings of IEEE Pacific Visualization Symposium 2009: 65 - 72, 2009

    @InProceedings{pvis09-grottel,
      author = {Grottel, S. and Reina, G. and Ertl, T.},
      title  = {{Optimized Data Transfer for Time-dependent, GPU-based Glyphs}},
      url    = {http://go.visus.uni-stuttgart.de/megamol},
      year   = {2009},
      pages  = {65-72},
      booktitle = {{Proceedings of IEEE Pacific Visualization Symposium 2009}}
    }

#
**MegaMol™ project website**  
[http://megamol.org](http://megamol.org)

    @misc{megamol,
      key  = "megamol",
      url  = {http://megamol.org},
      note = {{MegaMol project website \url{http://megamol.org}}},
    }