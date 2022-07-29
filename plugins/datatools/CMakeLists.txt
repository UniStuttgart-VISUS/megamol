# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(datatools
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd
    cluster
    geometry_calls
    mesh
  DEPENDS_EXTERNALS
    nanoflann
    sim_sort
    mmpld_io)

if (datatools_PLUGIN_ENABLED)
  find_package(ZeroMQ CONFIG REQUIRED)
  find_package(cppzmq CONFIG REQUIRED)
  find_package(imgui CONFIG REQUIRED)
  find_package(Eigen3 CONFIG REQUIRED)
  target_link_libraries(datatools
    PRIVATE
      libzmq
      cppzmq
      imgui::imgui
      eigen::target)

  # Additional sources
  file(GLOB_RECURSE extra_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/min_sphere_of_spheres/*.cpp")
  target_sources(datatools PRIVATE ${extra_source_files})
  target_include_directories(datatools PUBLIC "3rd/min_sphere_of_spheres")
  if (MPI_C_FOUND)
    target_link_libraries(datatools PRIVATE MPI::MPI_C)
  endif ()
endif ()
