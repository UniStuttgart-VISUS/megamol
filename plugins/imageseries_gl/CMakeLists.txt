# MegaMol
# Copyright (c) 2021-2023, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(imageseries_gl
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    imageseries
    mmstd
    mmstd_gl
  )
