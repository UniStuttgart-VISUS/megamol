# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmstd_datatools
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    geometry_calls
  DEPENDS_EXTERNALS
    tinyply
    Eigen
    nanoflann
    sim_sort
    mmpld_io
    libzmq
    libcppzmq)

if (mmstd_datatools_PLUGIN_ENABLED)
  # Additional sources
  file(GLOB_RECURSE extra_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/min_sphere_of_spheres/*.cpp")
  target_sources(mmstd_datatools PRIVATE ${extra_source_files})
  target_include_directories(mmstd_datatools PUBLIC "3rd/min_sphere_of_spheres")
  if (MPI_C_FOUND)
    target_link_libraries(mmstd_datatools PRIVATE MPI::MPI_C)
  endif ()
endif ()
