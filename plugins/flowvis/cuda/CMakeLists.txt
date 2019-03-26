#
# MegaMol™ flowvis Plugin
# Copyright 2019, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
if(BUILD_FLOWVIS_PLUGIN)
  project(flowvis_cuda)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cu")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cuh")

  # Find CUDA
  find_package(CUDA REQUIRED)
  cuda_include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
  link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
  include_directories(${CUDA_SDK_ROOT_DIR}/common/inc/)

  # Set architecture flags
  set(FLOWVIS_CUDA_FLAGS "-gencode arch=compute_61,code=sm_61" CACHE STRING "CUDA flags for setting, e.g., the used architecture")

  list(APPEND CUDA_NVCC_FLAGS ${FLOWVIS_CUDA_FLAGS})

  # Create CUDA library
  cuda_add_library(flowvis_streamlines_cuda ${source_files} ${header_files})

  # Installation rules for generated files
  if(WIN32)
    install(TARGETS flowvis_streamlines_cuda RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS flowvis_streamlines_cuda DESTINATION "lib")
  endif()

  # Grouping in Visual Studio
  set_target_properties(flowvis_streamlines_cuda PROPERTIES FOLDER plugins)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
endif()
