# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(molsurfmapcluster_gl
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    geometry_calls
    geometry_calls_gl
    protein_calls
    image_calls
    compositing_gl)

if (molsurfmapcluster_gl_PLUGIN_ENABLED)
  find_package(Eigen3 CONFIG REQUIRED)
  target_link_libraries(molsurfmapcluster_gl PRIVATE Eigen3::Eigen)
endif ()
