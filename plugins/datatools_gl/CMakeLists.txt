# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(datatools_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    geometry_calls_gl
    datatools
  DEPENDS_EXTERNALS
    tinyply)
