# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(archvis
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    mesh
    mesh_gl
    datatools)

#if (archvis_PLUGIN_ENABLED)
#  target_include_directories(archvis PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>)
#endif ()
