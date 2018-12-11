## Protein-CUDA

The Protein-CUDA plugin provides visualization and data processing modules for biomolecules. Contrary to the normal Protein plugin all of the modules contained here make use of CUDA.



## Build

This plugin depends on the Protein_Calls and the Geometry_Calls plugin. The plugin is switched OFF by default.

###### Linux

Currently, this plugin is only tested under Windows, so it will not appear in any Linux CMake configuration. If you want to test it there, please modify the plugins CMakeLists.txt file accordingly.

###### Windows

This plugin currently depends on CUDA with version 8 or greater. You have to have CUDA installed on your system in order to make this plugin detectable by CMake. If you have an earlier CUDA version installed, CMake will notify you.



## Modules

### QuickSurfRenderer

Renderer for a Gaussian Density Surface for a `MolecularDataCall`.  It first splats all atoms into a volume and then performs the Marching Cubes algorithm to reconstruct an isosurface. The resulting mesh then gets rendered

### QuickSurfRenderer2

Practically the same as the `QuickSurfRenderer`, but working for a `MultiParticleDataCall`.

### QuickSurfRaycaster

Does the the same as the `QuickSurfRenderer`, but leaves out the reconstruction of the Mesh via Marching Cubes. Instead, the isosurface is renderer via Raycasting. Additionally the Module is capable of sending out the raycasted volume to other Modules.

### SecStructFlattener

Performs a re-layouting of the atom positions of a protein, in order to achieve more appealing results when using a `SecStructRenderer2D`. This module is using force-based graph layouting techniques to move the atoms around.  

### SecStructRenderer2D

Renders the secondary structure of a protein in a two-dimensional manner by flattening the geometry onto a plane. Should only be used with a preceding `SecStructFlattener`.

### ComparativeMolSurfaceRenderer

Performs a comparison of two different biomolecules by warping the geometry of one onto the other.

### Other Modules

TODO
