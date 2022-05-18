# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mesh
  BUILD_DEFAULT ON
  DEPENDS_EXTERNALS PUBLIC
    quickhull
    tinyobjloader
    tinygltf
    obj-io)

if (mesh_PLUGIN_ENABLED)
  if (MSVC)
    target_compile_definitions(mesh PRIVATE NOMINMAX)
  endif ()

  # Find CGAL
  find_package(CGAL REQUIRED)

  if(NOT TARGET CGAL::CGAL)
    message(FATAL_ERROR "Target for CGAL not found")
  endif()

  if (TARGET CGAL)
    set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
  endif()

  target_link_libraries(mesh PRIVATE CGAL::CGAL)
endif ()
