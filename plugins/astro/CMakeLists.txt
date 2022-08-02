# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(astro
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    datatools
    geometry_calls)

if (astro_PLUGIN_ENABLED)
  find_package(nanoflann CONFIG REQUIRED)
  find_package(simultaneous_sort CONFIG REQUIRED)
  target_link_libraries(astro
    PRIVATE
      nanoflann::nanoflann
      simultaneous_sort)
endif()
