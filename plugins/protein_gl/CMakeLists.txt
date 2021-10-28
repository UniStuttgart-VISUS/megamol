# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(protein_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    protein
    mmstd_trisoup
    protein_calls
    geometry_calls
  DEPENDS_EXTERNALS
    glad
    nanoflann)

