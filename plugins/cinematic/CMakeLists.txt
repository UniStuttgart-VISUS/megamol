# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(cinematic
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd)

if (cinematic_PLUGIN_ENABLED)
  find_package(imgui CONFIG REQUIRED)
  find_package(PNG REQUIRED)

  target_link_libraries(cinematic
    PRIVATE
      imgui::imgui
      PNG::PNG)
endif ()
