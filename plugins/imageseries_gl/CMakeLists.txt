# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(imageseries_gl
  BUILD_DEFAULT OFF
  DEPENDS_GL
  DEPENDS_PLUGINS
    imageseries
  DEPENDS_EXTERNALS
  	glowl
  )

if (imageseries_gl_PLUGIN_ENABLED)
endif ()
