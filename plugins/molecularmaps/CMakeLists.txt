# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(molecularmaps
  BUILD_DEFAULT OFF
  DEPENDS_CUDA
  DEPENDS_PLUGINS
    protein_calls
    geometry_calls
  DEPENDS_EXTERNALS
    Eigen)

if (molecularmaps_PLUGIN_ENABLED)

  set(CMAKE_CUDA_ARCHITECTURES 61)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

  get_filename_component(cuda_compiler_directory "${CMAKE_CUDA_COMPILER}" DIRECTORY)

  file(GLOB_RECURSE cuda_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cu")
  file(GLOB_RECURSE cuda_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cuh")
  file(GLOB_RECURSE cuda_helpers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/helper_includes/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

  # Set language to CUDA for respective files.
  set_source_files_properties(cuda_source_files PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(source_files PROPERTIES LANGUAGE CPP)

  # Target definition
  target_sources(molecularmaps PRIVATE ${cuda_header_files} ${cuda_source_files})
  set_target_properties(molecularmaps PROPERTIES CUDA_STANDARD 17)

  target_include_directories(molecularmaps PUBLIC "src/helper_includes")
  target_link_libraries(molecularmaps PRIVATE glad)

  if(NOT MSVC)
    get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
    target_include_directories(molecularmaps PUBLIC "${CUDA_BIN_DIR}/../include")
    target_compile_definitions(molecularmaps PRIVATE __CUDACC__)
  endif()

  # Grouping in Visual Studio
  source_group("CUDA Header Files" FILES ${cuda_header_files})
  source_group("CUDA Source Files" FILES ${cuda_source_files})
endif ()
