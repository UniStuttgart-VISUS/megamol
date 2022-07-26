# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(volume_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    datatools
    volume
    geometry_calls
  DEPENDS_EXTERNALS
    nanoflann
    sim_sort
    glowl)
