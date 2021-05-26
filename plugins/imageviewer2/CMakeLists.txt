# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(imageviewer2
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    image_calls)

if (imageviewer2_PLUGIN_ENABLED)
  if (UNIX)
    find_package(JPEG REQUIRED)
    target_link_libraries(imageviewer2 PRIVATE JPEG::JPEG)
  endif ()

  if (MPI_C_FOUND)
    target_link_libraries(imageviewer2 PRIVATE MPI::MPI_C)
  endif ()
endif ()
