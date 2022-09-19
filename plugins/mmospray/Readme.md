# MegaMol Plugin: OSPRay

[OSPRay](http://ospray.org) is a CPU ray tracing engine and is one project of the Software Defined Visualization (SDVis) open source initiative of Intel (http://sdvis.org/).
It builds on top of the high-performance ray tracing kernels of [Embree](https://embree.github.io/) and the [ISPC](https://ispc.github.io/) SPMD compiler.

In this plugin, most of the functionality of OSPRay is covered and can be used via MegaMol modules and MegaMol calls.
This plugin supports the chain paradigm that allows lights and structures to be stacked arbitrarily.
The figure below shows a common OSPRay module call graph in MegaMol.

![](ospray_configurator.png)


## Building

[OSPRay](http://ospray.org) is pulled via vcpkg, you can just switch on `MEGAMOL_USE_OSPRAY` and then enable this plugin.

## Modules

As seen in the figure above, the OSPRay plugin has three different kinds of modules: `OSPRayStructure`, `OSPRayLight`, and  `OSPRayMaterial`.
While these three modules are processing the actual data and several parameters, main module of this plugin is the `OSPRayRenderer` that communicats to OSPRay via its API.
