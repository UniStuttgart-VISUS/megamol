# MegaMol module for Advanced Ambient Occlusion
This module allows the rendering of particle data with an advanced ambient occlusion algorithm, based on approximate voxel cone tracing ([Original Paper by Crassin](http://maverick.inria.fr/Publications/2011/CNSGE11b/)). In case you do not know what MegaMol is, please visit [The MegaMol Website](http://megamol.org).

![random sphere data set with purely ambient occlusion shading](https://github.com/jstaib-tud/megamol-mdao2/raw/master/demo.png)

The plugin provides the module `MDAO2Renderer`. It has the following features:
* Smooth and great looking ambient occlusion based on voxelization and cone tracing
* Quite fast through deferred shading and voxelization that is nearly purely evaluated on the GPU
* Support for multiple particle lists and list types
* Works together with the clipping plane module and transfer function module
* Uses modern features of OpenGL 4.5., but can fall back to OpenGL 3.3.

The general workflow of the algorithm is:

1. Generate a coarse voxelization of the scene where each voxel represents a density of spheres in the spacial cell (see [this paper by Grottel et al.](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6183593) for more explanations on this concept).
2. Render the scene's geometry and store the normals, color and depth of visible fragments in buffers for deferred shading
3. Render the visible fragments and for each fragment:
  * Evaluate local Phong lighting
  * Evaluate ambient occlusion by tracing 3 approximate cones, tightly arranged around the normal in the voxelized scene, gathering the densities of the spheres
  * Weight the local lighting with the ambient occlusion term

## Building

Remember to git clone recursive. Otherwise, you need to fetch the submodules as well.

To run this plugin, you need:
* Linux or Windows
* An installation of MegaMol (at least version 1.1), see [here](https://svn.vis.uni-stuttgart.de/trac/megamol/wiki/HowToBuild11) how to build one
* OpenGL, at least version 3.3

The build process is straight forward and described (for another plugin) on the [MegaMol building page](https://svn.vis.uni-stuttgart.de/trac/megamol/wiki/HowToBuild11), secton "Plugins".

## Parameters
The module MDAO2Renderer exposes the following parameters. The values in brackets indicate the default values:
* `enable_lighting` (`false`): Shade the spheres using local Phong lighting. If turned off, a constant color is used
* `enable_ao` (`true`): Actually enable ambient occlusion
* `ao_volsize` (`128`): Size of the longest edge of the volume that holds the voxelized scene. The actual volume extent corresponds to the extent of the scene's clip box.
* `ao_apex` (`50`): Aperture of the cone (should be renamed) in degrees. 
* `ao_offset` (`0.01`): Offset of the cones from the surface point in object space units in order not to gather occlusion from the own volume.
* `ao_strength` (`1.0`): Multiplicator to strengthen or weaken the effect of ambient occlusion
* `ao_conelen` (`0.8`): Length of the cones in object space units.
* `high_prec_tex` (`false`): Use a high precision texture to store the normals for deferred shading. Might decrease artifacts from lighting but reduces performance.
