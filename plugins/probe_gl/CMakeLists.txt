# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

find_package(imgui CONFIG REQUIRED)

megamol_plugin(probe_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    datatools
    probe
    mesh
    mesh_gl
    compositing_gl
  DEPENDS_LIBRARIES PUBLIC
    imgui::imgui)
