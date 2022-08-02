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
    geometry_calls)

if (protein_PLUGIN_ENABLED)
  find_package(chemfiles CONFIG REQUIRED)
  target_link_libraries(protein
    PRIVATE
      vislib
      chemfiles)
endif ()
