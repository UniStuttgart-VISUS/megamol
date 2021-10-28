# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(infovis_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd_datatools
    compositing_gl
  DEPENDS_EXTERNALS
    Delaunator)
