# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(volume
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    geometry_calls)

if (volume_PLUGIN_ENABLED)
  #XXX: hacky appraoch to include datraw
  add_subdirectory(datraw)

  target_link_libraries(volume PRIVATE datraw)
endif ()
