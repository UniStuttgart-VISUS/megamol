# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(infovis
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd
    datatools)

if (infovis_PLUGIN_ENABLED)
  find_package(Eigen3 CONFIG REQUIRED)
  find_package(bhtsne CONFIG REQUIRED)
  find_package(ltla_umappp CONFIG REQUIRED)
  find_path(DELAUNATOR_CPP_INCLUDE_DIRS "delaunator.hpp")

  target_link_libraries(infovis
    PRIVATE
      Eigen3::Eigen
      bhtsne::bhtsne
      ltla::umappp)
  target_include_directories(infovis
    PRIVATE
      ${DELAUNATOR_CPP_INCLUDE_DIRS})

  if (MSVC)
    # Avoid C3016 due to modern OpenMP usage. Note: /openmp:llvm might work as well.
    # Avoid C1128 due to excessive inlining.
    set_source_files_properties("src/UMAProjection.cpp" PROPERTIES COMPILE_FLAGS "/openmp- /bigobj")
  endif ()

  # Additional sources
  file(GLOB_RECURSE extra_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/*.h")
  file(GLOB_RECURSE extra_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/*.cpp")
  target_sources(infovis PRIVATE ${extra_header_files} ${extra_source_files})
  target_include_directories(infovis PRIVATE "3rd")
endif ()
