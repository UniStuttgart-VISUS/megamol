# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(infovis
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd_datatools
  DEPENDS_EXTERNALS
    Eigen
    Delaunator
    nanoflann
    bhtsne)

if (infovis_PLUGIN_ENABLED)
  # Additional sources
  file(GLOB_RECURSE extra_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/*.h")
  file(GLOB_RECURSE extra_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/*.cpp")
  target_sources(infovis PRIVATE ${extra_header_files} ${extra_source_files})
  target_include_directories(infovis PRIVATE "3rd")

  # Format
  file(GLOB_RECURSE all_cpp_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
  add_clang_format(infovis
    STYLE "${CMAKE_SOURCE_DIR}/.clang-format"
    FILES ${all_cpp_files})
endif ()
