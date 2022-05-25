# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mesh_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd_gl
    mesh
    compositing_gl)
