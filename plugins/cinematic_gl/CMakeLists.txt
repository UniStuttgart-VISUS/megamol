# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(cinematic_gl
  BUILD_DEFAULT ON
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    cinematic)

if (cinematic_gl_PLUGIN_ENABLED)
  find_package(imgui CONFIG REQUIRED)
  find_package(PNG REQUIRED)

  target_link_libraries(cinematic_gl
    PRIVATE
      imgui::imgui
      PNG::PNG)
endif ()
