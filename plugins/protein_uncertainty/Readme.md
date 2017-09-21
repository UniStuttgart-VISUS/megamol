## Protein_Uncertainty-Plugin

### About

The Protein_Uncertainty-Plugin contains modules that are targeted on the visualization of the uncertainty in protein data.

### Prerequisites
* VISLib
* MegaMol-Core
* protein_calls-Plugin
* protein-Plugin
* mmstd_trisoup-Plugin

### Using the Plugin
This section introduces some of the more important modules and calls of the Protein-Plugin.

| Module | Description |
|:-------|:------------|
| PDBLoader | Loads Protein Data Bank (.pdb) files. Normally, these contain atom coordinates and additional meta-information. A detailed specification of the file format can be found at the [site of the wwPDB](http://www.wwpdb.org/documentation/file-format "http://www.wwpdb.org/documentation/file-format"). Note, that this module only supports the more frequently used fields of the format.|
| BindingSiteDataSource | Expects a file containing a binding site description for proteins. This description is similar to the PDB binding site description. |
| MoleculeBallifier | Converts atom data stored inside a MolecularDataCall to generic particle data stored in a MultiParticleDataCall. Allows the usage of the more generic renderers for protein data. |
| MoleculeCartoonRenderer | Offers rendering of the Cartoon-representation of a protein. |
| MoleculeSequenceRenderer | Offers 2D sequence rendering for protein data. |
| CartoonTessellationRenderer | A specialized renderer for the Cartoon-representation of a protein using the more recent capabilities of the graphics hardware. Has not as many features as the MoleculeCartoonRenderer|
| SimpleMoleculeRenderer | Offers direct rendering of protein data in ball- or ball&stick-representation. |
| View3DMouse | A version of the View3D contained in the core with additional mouse callback support.|
| View3DSpaceMouse| Same as the View3DMouse, but with support for a spacemouse.|