# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(archvis_gl
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    mesh
    mesh_gl
    datatools)

if (archvis_gl_PLUGIN_ENABLED)
  find_package(quickhull CONFIG REQUIRED)
  target_link_libraries(archvis_gl
    PRIVATE
      quickhull::quickhull)
endif ()
