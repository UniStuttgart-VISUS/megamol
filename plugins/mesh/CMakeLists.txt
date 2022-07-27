# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mesh
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd
  DEPENDS_EXTERNALS PUBLIC
    quickhull
    tinygltf
    obj-io)

find_package(tinyobjloader)

target_link_libraries(mesh PUBLIC tinyobjloader::tinyobjloader)

if (mesh_PLUGIN_ENABLED)
  if (MSVC)
    target_compile_definitions(mesh PRIVATE NOMINMAX)
  endif ()

  # Use CGAL
  if (MEGAMOL_USE_CGAL)
    target_link_libraries(mesh PRIVATE CGAL::CGAL)
  endif ()
endif ()
