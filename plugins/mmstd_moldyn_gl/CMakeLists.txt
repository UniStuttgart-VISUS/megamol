# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmstd_moldyn_gl
  BUILD_DEFAULT ON 
  DEPENDS_GL
  DEPENDS_PLUGINS 
    mmstd_moldyn 
    geometry_calls)
