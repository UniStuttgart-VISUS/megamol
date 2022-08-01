# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(volume_gl
  BUILD_DEFAULT ON
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    datatools
    volume
    geometry_calls
  DEPENDS_EXTERNALS
    glowl)

if (volume_gl_PLUGIN_ENABLED)
  find_package(simultaneous_sort CONFIG REQUIRED)
  target_link_libraries(volume_gl
    PRIVATE
      simultaneous_sort)
endif()
