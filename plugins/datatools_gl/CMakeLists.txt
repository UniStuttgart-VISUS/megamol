# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(datatools_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    geometry_calls_gl
    datatools
    imgui
  DEPENDS_EXTERNALS
    tinyply)
