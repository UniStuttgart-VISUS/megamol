# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmospray
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    geometry_calls
    datatools
    protein_calls
    mesh)

if (mmospray_PLUGIN_ENABLED)

  find_package(ospray CONFIG REQUIRED)

  target_link_libraries(mmospray PUBLIC ospray::ospray ospray::ospray_sdk ospray::ospray_module_cpu)

endif ()
