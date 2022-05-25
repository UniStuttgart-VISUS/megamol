# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(moldyn_gl
  BUILD_DEFAULT ON 
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd_gl
    moldyn
    datatools
    geometry_calls
    imgui)
