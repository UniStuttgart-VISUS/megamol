# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmospray_gl
  BUILD_DEFAULT ON
  DEPENDS_FEATURES
    opengl
    ospray
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    mmospray
    mesh
    geometry_calls_gl)
