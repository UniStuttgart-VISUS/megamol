# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

find_package(imgui CONFIG REQUIRED)
find_package(PNG REQUIRED)

megamol_plugin(cinematic
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd
  DEPENDS_LIBRARIES
    imgui::imgui
    PNG::PNG)
