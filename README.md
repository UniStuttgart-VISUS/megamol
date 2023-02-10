
![](docs/images/logo.png)

[![Build Status - Azure DevOps Server at VIS[US]][build-button]][build-link]
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.][project-button]][project-link]
[![License: BSD-3-Clause][license-button]][license-link]
[![Recent Commit Activity][commit-button]][commit-link]

[build-button]: https://img.shields.io/github/checks-status/UniStuttgart-VISUS/megamol/master?label=Azure%20Pipelines&logo=Azure%20Pipelines
[build-link]: https://tfs.visus.uni-stuttgart.de/tfs/VIS(US)/MegaMol/_build/latest?definitionId=32&branchName=master
[project-button]: https://www.repostatus.org/badges/latest/active.svg
[project-link]: https://www.repostatus.org/#active
[license-button]: https://img.shields.io/github/license/UniStuttgart-VISUS/megamol
[license-link]: LICENSE
[commit-button]: https://img.shields.io/github/commit-activity/m/UniStuttgart-VISUS/megamol
[commit-link]: https://github.com/UniStuttgart-VISUS/megamol/commits/master

MegaMol is a visualization middleware used to visualize point-based molecular data sets.
This software is developed within the Collaborative Research Center 716, subproject D.3 at the [Visualization Research Center (VISUS)](https://www.visus.uni-stuttgart.de/en) of the University of Stuttgart and at the Computer Graphics and Visualization Group of the TU Dresden.

MegaMol succeeds MolCloud, which has been developed at the University of Stuttgart in order to visualize point-based data sets.
MegaMol is written in C++, and uses an OpenGL as Rendering-API and GLSL-Shader.
It supports the operating systems Microsoft Windows and Linux, each in 64-bit version.
In large parts, MegaMol is based on VISlib, a C++-class library for scientific visualization, which has also been developed at the University of Stuttgart.

## Manual

See the [manual](docs/manual.md) for detailed instructions on how to build and use MegaMol.

## License

See the [license](LICENSE) file.

## Citing MegaMol

Please use one of the following methods to reference the MegaMol project.

**MegaMol – A Comprehensive Prototyping Framework for Visualizations**  
P. Gralka, M. Becher, M. Braun, F. Frieß, C. Müller, T. Rau, K. Schatz, C. Schulz, M. Krone, G. Reina, T. Ertl  
The European Physical Journal Special Topics, vol. 227, no. 14, pp. 1817-1829, 2019  
DOI: [10.1140/epjst/e2019-800167-5](https://doi.org/10.1140/epjst/e2019-800167-5)
<details>
  <summary>BibTeX</summary>

  ```bibtex
  @Article{Gralka2019MegaMol,
    author    = {Patrick Gralka and Michael Becher and Matthias Braun and Florian Frie{\ss} and Christoph M{\"u}ller and Tobias Rau and Karsten Schatz and Christoph Schulz and Michael Krone and Guido Reina and Thomas Ertl},
    journal   = {The European Physical Journal Special Topics},
    title     = {{MegaMol} {\textendash} a comprehensive prototyping framework for visualizations},
    year      = {2019},
    month     = {mar},
    number    = {14},
    pages     = {1817--1829},
    volume    = {227},
    issn      = {1951-6401},
    doi       = {10.1140/epjst/e2019-800167-5},
    publisher = {Springer Science and Business Media {LLC}},
  }
  ```
</details>

**MegaMol – A Prototyping Framework for Particle-based Visualization**  
S. Grottel, M. Krone, C. Müller, G. Reina, T. Ertl  
IEEE Transactions on Visualization and Computer Graphics, vol. 21, no. 2, pp. 201-214, 2015  
DOI: [10.1109/TVCG.2014.2350479](https://doi.org/10.1109/TVCG.2014.2350479)
<details>
  <summary>BibTeX</summary>

  ```bibtex
  @Article{Grottel2015MegaMol,
    author    = {Sebastian Grottel and Michael Krone and Christoph M{\"u}ller and Guido Reina and Thomas Ertl},
    journal   = {{IEEE} Transactions on Visualization and Computer Graphics},
    title     = {{MegaMol}{\textemdash}A Prototyping Framework for Particle-Based Visualization},
    year      = {2015},
    month     = {feb},
    number    = {2},
    pages     = {201--214},
    volume    = {21},
    issn      = {1077-2626}
    doi       = {10.1109/tvcg.2014.2350479},
    publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
  }
  ```
</details>

**Coherent Culling and Shading for Large Molecular Dynamics Visualization**  
S. Grottel, G. Reina, C. Dachsbacher, T. Ertl  
Computer Graphics Forum, vol. 29, no. 3, pp. 953-962, 2010  
DOI: [10.1111/j.1467-8659.2009.01698.x](https://doi.org/10.1111/j.1467-8659.2009.01698.x)
<details>
  <summary>BibTeX</summary>

  ```bibtex
  @Article{Grottel2010MegaMol,
    author    = {Sebastian Grottel and Guido Reina and Carsten Dachsbacher and Thomas Ertl},
    journal   = {Computer Graphics Forum},
    title     = {Coherent Culling and Shading for Large Molecular Dynamics Visualization},
    year      = {2010},
    month     = {aug},
    number    = {3},
    pages     = {953--962},
    volume    = {29},
    doi       = {10.1111/j.1467-8659.2009.01698.x},
    publisher = {Wiley},
  }
  ```
</details>

**Optimized Data Transfer for Time-dependent, GPU-based Glyphs**  
S. Grottel, G. Reina, T. Ertl  
In Proceedings of IEEE Pacific Visualization Symposium 2009: 65 - 72, 2009  
DOI: [10.1109/PACIFICVIS.2009.4906839](https://doi.org/10.1109/PACIFICVIS.2009.4906839)
<details>
  <summary>BibTeX</summary>

  ```bibtex
  @InProceedings{Grottel2009MegaMol,
    author    = {S. Grottel and G. Reina and T. Ertl},
    booktitle = {2009 {IEEE} Pacific Visualization Symposium},
    title     = {Optimized data transfer for time-dependent, {GPU}-based glyphs},
    year      = {2009},
    month     = {apr},
    pages     = {65--72},
    doi       = {10.1109/pacificvis.2009.4906839},
  }
  ```
</details>

**MegaMol project website**  
[https://megamol.org](https://megamol.org)
<details>
  <summary>BibTeX</summary>

  ```bibtex
  @Misc{MegaMolWebsite,
    title        = {{MegaMol project website}},
    howpublished = {\url{https://megamol.org}},
  }
  ```
</details>
