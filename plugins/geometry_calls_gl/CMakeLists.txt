# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(geometry_calls_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd_gl
    PUBLIC geometry_calls)
