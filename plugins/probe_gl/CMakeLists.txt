# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(probe_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd_gl
    datatools
    probe
    mesh
    mesh_gl
    compositing_gl
  DEPENDS_EXTERNALS PUBLIC
    imgui)
