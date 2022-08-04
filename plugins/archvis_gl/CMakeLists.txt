# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(archvis_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    mesh
    mesh_gl
    datatools)

#if (archvis_PLUGIN_ENABLED)
#  target_include_directories(archvis PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>)
#endif ()
