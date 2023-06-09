# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(test_gl
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    # feature1
  DEPENDS_PLUGINS
    mmstd
    # plugin1
    # plugin2
  )

if (test_gl_PLUGIN_ENABLED)
  # Do additional plugin specific stuff, not covered by the generic plugin creation,
  # i.e. find_package() dependencies, additional sources 'target_sources(megamolplugin PRIVATE ...)', ....
endif ()
