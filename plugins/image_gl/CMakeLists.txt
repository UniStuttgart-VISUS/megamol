# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(image_gl
  BUILD_DEFAULT ON
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    cluster
    image_calls)

if (image_gl_PLUGIN_ENABLED)
  if (UNIX)
    find_package(JPEG REQUIRED)
    target_link_libraries(image_gl PRIVATE JPEG::JPEG)
  endif ()

  if (MPI_C_FOUND)
    target_link_libraries(image_gl PRIVATE MPI::MPI_C)
  endif ()
endif ()
