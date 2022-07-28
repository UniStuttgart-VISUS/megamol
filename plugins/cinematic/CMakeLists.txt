# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

find_package(PNG REQUIRED)

megamol_plugin(cinematic
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd
  DEPENDS_LIBRARIES
    PNG::PNG)
