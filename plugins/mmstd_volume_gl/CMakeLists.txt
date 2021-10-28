# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmstd_volume_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_EXTERNALS
    glowl 
  DEPENDS_PLUGINS
    mmstd_volume
    geometry_calls)
