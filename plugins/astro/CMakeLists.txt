# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(astro
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    datatools
    geometry_calls
  DEPENDS_EXTERNALS
    nanoflann
    glowl)

if (astro_PLUGIN_ENABLED)
  find_package(simultaneous_sort CONFIG REQUIRED)
  target_link_libraries(astro PRIVATE simultaneous_sort)
endif()
