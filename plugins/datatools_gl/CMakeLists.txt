# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(datatools_gl
  BUILD_DEFAULT ON
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    geometry_calls_gl
    datatools)

if (datatools_gl_PLUGIN_ENABLED)
  find_package(imgui CONFIG REQUIRED)
  find_package(tinyply CONFIG REQUIRED)
  target_link_libraries(datatools_gl
    PRIVATE
      imgui::imgui
      tinyply)
endif ()
