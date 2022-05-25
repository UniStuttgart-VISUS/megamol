# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(trisoup_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd_gl
    trisoup
    geometry_calls_gl
    compositing_gl)
