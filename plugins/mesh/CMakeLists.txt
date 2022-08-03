# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mesh
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd
  DEPENDS_EXTERNALS PUBLIC
    quickhull)

if (mesh_PLUGIN_ENABLED)
  find_package(tinyobjloader CONFIG REQUIRED)
  find_path(OBJ_IO_INCLUDE_DIRS "obj_io.h")
  find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")

  target_link_libraries(mesh
    PRIVATE
      tinyobjloader::tinyobjloader)
  target_include_directories(mesh
    PRIVATE
      ${OBJ_IO_INCLUDE_DIRS}
      ${TINYGLTF_INCLUDE_DIRS})

  if (MSVC)
    target_compile_definitions(mesh PRIVATE NOMINMAX)
  endif ()

  # Use CGAL
  if (MEGAMOL_USE_CGAL)
    target_link_libraries(mesh PRIVATE CGAL::CGAL)
  endif ()
endif ()
