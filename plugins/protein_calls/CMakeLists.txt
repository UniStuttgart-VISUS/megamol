# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(protein_calls
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd)

if (protein_calls_PLUGIN_ENABLED)
  find_package(Eigen3 CONFIG REQUIRED)
  target_link_libraries(protein_calls
    PRIVATE
      eigen::target)
endif ()
