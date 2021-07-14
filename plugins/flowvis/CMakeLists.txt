# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(flowvis
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mesh)

if (flowvis_PLUGIN_ENABLED)
  # Find CGAL
  find_package(CGAL REQUIRED)

  if(NOT TARGET CGAL::CGAL)
    message(FATAL_ERROR "Target for CGAL not found")
  endif()

  if (TARGET CGAL)
    set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
  endif()

  target_link_libraries(flowvis PRIVATE CGAL::CGAL)
endif ()
