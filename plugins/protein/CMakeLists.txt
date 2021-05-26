# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(protein
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd_trisoup
    protein_calls
    geometry_calls
  DEPENDS_EXTERNALS
    glad
    nanoflann)

if (protein_PLUGIN_ENABLED)
  target_link_libraries(protein PRIVATE vislib)

  if (WIN32)
    # special case for the MSMS-related files
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/msms/msms.exe DESTINATION "bin")
  endif ()
endif ()
