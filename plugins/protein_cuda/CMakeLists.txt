# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(protein_cuda
  BUILD_DEFAULT OFF
  DEPENDS_CUDA
  DEPENDS_PLUGINS
    geometry_calls
    geometry_calls_gl
    protein_calls)

if (protein_cuda_PLUGIN_ENABLED)

  if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
    message(FATAL_ERROR "Protein CUDA currently only supports CUDA Version 11 or above. The version found was CUDA ${CMAKE_CUDA_COMPILER_VERSION}. Please turn off the Protein CUDA plugin or install a correct version of CUDA.")
  endif ()

  set(CMAKE_CUDA_ARCHITECTURES 52)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

  get_filename_component(CUDA_COMPILER_DIRECTORY "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)

  # Suppress CUDA warnings
  if (SUPPRESS_CUDA_WARNINGS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"-w\"")
  endif ()

  # Collect source files
  file(GLOB_RECURSE cuda_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cu")
  file(GLOB_RECURSE cuda_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cuh")
  file(GLOB cuda_cufft_binaries "${CUDA_COMPILER_DIRECTORY}/cufft*.dll")

  # Set language to CUDA for respective files
  set_source_files_properties(cuda_source_files PROPERTIES LANGUAGE CUDA)

  # Remove special files from the build
  list(REMOVE_ITEM cuda_source_files "src/cuenergy.cu")
  list(REMOVE_ITEM cuda_source_files "src/gpu_poisson_solver.cu")
  list(REMOVE_ITEM cuda_source_files "src/particles_kernel.cu")

  # Target definition
  target_sources(protein_cuda PRIVATE ${cuda_header_files} ${cuda_source_files})
  set_target_properties(protein_cuda PROPERTIES CUDA_STANDARD 17)
  target_include_directories(protein_cuda PUBLIC "src/helper_includes" "${CUDA_BIN_DIR}/../include")
  target_link_libraries(protein_cuda PRIVATE cufft)

  # Install cufft (this seems to be necessary for every needed extra CUDA library)
  install(FILES ${cuda_cufft_binaries} DESTINATION "bin")

  # Grouping in Visual Studio
  source_group("CUDA Header Files" FILES ${cuda_header_files})
  source_group("CUDA Source Files" FILES ${cuda_source_files})
endif ()
