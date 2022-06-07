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

  # Use CGAL
  if (ENABLE_CGAL)
    target_link_libraries(mesh PRIVATE CGAL::CGAL)
  endif ()
endif ()
