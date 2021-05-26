# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mesh
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    compositing_gl
  DEPENDS_EXTERNALS PUBLIC
    quickhull
    tinyobjloader
    tinygltf)

if (mesh_PLUGIN_ENABLED)
  if (MSVC)
    target_compile_definitions(mesh PRIVATE NOMINMAX)
  endif ()
endif ()
