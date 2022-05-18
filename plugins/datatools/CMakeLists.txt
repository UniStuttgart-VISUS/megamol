# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(datatools
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS PUBLIC
    geometry_calls
    mesh
  DEPENDS_EXTERNALS
    Eigen
    nanoflann
    sim_sort
    mmpld_io
    libzmq
    libcppzmq
    imgui)

if (datatools_PLUGIN_ENABLED)
  # Additional sources
  file(GLOB_RECURSE extra_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/min_sphere_of_spheres/*.cpp")
  target_sources(datatools PRIVATE ${extra_source_files})
  target_include_directories(datatools PUBLIC "3rd/min_sphere_of_spheres")
  if (MPI_C_FOUND)
    target_link_libraries(datatools PRIVATE MPI::MPI_C)
  endif ()
endif ()
