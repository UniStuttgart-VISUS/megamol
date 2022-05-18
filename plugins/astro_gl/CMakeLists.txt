# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(astro_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    datatools
    geometry_calls
  DEPENDS_EXTERNALS
    nanoflann
    sim_sort
    glowl)
