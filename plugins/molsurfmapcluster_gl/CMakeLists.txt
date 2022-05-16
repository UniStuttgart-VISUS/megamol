# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(molsurfmapcluster_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    geometry_calls
    geometry_calls_gl
    protein_calls
    image_calls
  DEPENDS_EXTERNALS
    Eigen)
