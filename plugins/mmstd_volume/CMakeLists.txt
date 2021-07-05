# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmstd_volume
  BUILD_DEFAULT ON
  DEPENDS_EXTERNALS
    glowl)

if (mmstd_volume_PLUGIN_ENABLED)
  #XXX: hacky appraoch to include datraw
  add_subdirectory(datraw)

  target_link_libraries(mmstd_volume PRIVATE datraw)
endif ()
