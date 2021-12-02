# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(protein_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    protein
    trisoup
    protein_calls
    geometry_calls_gl
    compositing_gl
  DEPENDS_EXTERNALS
    glad
    nanoflann)

