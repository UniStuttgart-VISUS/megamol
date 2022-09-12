# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(probe_gl
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    datatools
    probe
    mesh
    mesh_gl
    compositing_gl)

if (probe_gl_PLUGIN_ENABLED)
  find_package(imgui CONFIG REQUIRED)
  target_link_libraries(probe_gl
    PRIVATE
      imgui::imgui)
endif ()
