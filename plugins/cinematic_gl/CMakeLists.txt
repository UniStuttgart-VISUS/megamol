# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

find_package(imgui CONFIG REQUIRED)
find_package(PNG REQUIRED)

megamol_plugin(cinematic_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    cinematic
  DEPENDS_LIBRARIES
    imgui::imgui
    PNG::PNG)
