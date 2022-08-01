# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(protein
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    trisoup
    protein_calls
    geometry_calls
  DEPENDS_EXTERNALS
    chemfiles)

if (protein_PLUGIN_ENABLED)
  target_link_libraries(protein
    PRIVATE
      vislib)
endif ()
