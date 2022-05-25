# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmospray_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd_gl
    mmospray
    mesh
    geometry_calls_gl)
