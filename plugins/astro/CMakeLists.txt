# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(astro
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd_datatools
  DEPENDS_EXTERNALS
    nanoflann
    sim_sort
    glowl)
