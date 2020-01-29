#
# MegaMol™ flowvis Plugin
# Copyright 2019, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
if(BUILD_FLOWVIS_PLUGIN)
  project(flowvis_cuda LANGUAGES CXX CUDA)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cu")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cuh")

  set_source_files_properties(source_files PROPERTIES LANGUAGE CUDA)

  # Suppress warnings
  if(SUPPRESS_CUDA_WARNINGS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"-w\"")
  endif()

  # Set fPIC
  if(NOT WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"-fPIC\"")
  endif()

  # Create CUDA library
  add_library(flowvis_streamlines_cuda STATIC ${source_files} ${header_files})

  set_target_properties(flowvis_streamlines_cuda PROPERTIES CUDA_STANDARD 14)

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
