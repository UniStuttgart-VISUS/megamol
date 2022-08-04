# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(megamolplugin
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    # plugin1
    # plugin2
  DEPENDS_EXTERNALS
    # external1
    # external2
  )

if (megamolplugin_PLUGIN_ENABLED)
  # Do additional plugin specific stuff, not covered by the generic plugin creation,
  # i.e. find_package() dependencies, additional sources 'target_sources(megamolplugin PRIVATE ...)', ....
endif ()
