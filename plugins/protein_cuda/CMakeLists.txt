#
# MegaMol™ protein_cuda Plugin
# Copyright 2015, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
include(CheckLanguage)

check_language(CUDA)
if(MSVC AND CMAKE_CUDA_COMPILER)
  option(BUILD_PROTEIN_CUDA_PLUGIN "Option to build protein_cuda" OFF)
else()
  set(BUILD_PROTEIN_CUDA_PLUGIN OFF)
endif()

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(BUILD_PROTEIN_CUDA_PLUGIN)
  project(protein_cuda LANGUAGES CXX CUDA)

  string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
  if(NOT CUDA_VERSION_MAJOR GREATER 7)
    message(FATAL_ERROR "Protein_CUDA currently only supports CUDA Version 8 or above. The version found was CUDA ${CUDA_VERSION_MAJOR} (${CMAKE_CUDA_COMPILER_VERSION}). Please turn off the Protein_CUDA plugin or install a correct version of CUDA." )
  endif()

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_PROTEIN_CUDA_PLUGIN BUILD_GEOMETRY_CALLS_PLUGIN BUILD_PROTEIN_CALLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  #NVCC options
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

  get_filename_component(CUDA_COMPILER_DIRECTORY  "${CMAKE_CUDA_COMPILER}" DIRECTORY)

  if(MSVC)
    include_directories("include" "src" "src/helper_includes")
  else()
    get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
    include_directories("include" "src" "src/helper_includes" ${CUDA_BIN_DIR}/../include)
    add_definitions(-D__CUDACC__)
  endif()
  
  # Build MegaMol Plugin
  # public header files for installation
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  # source files for compilation
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")
  # CUDA related files
  file(GLOB_RECURSE cuda_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cu")
  file(GLOB_RECURSE cuda_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cuh")
  file(GLOB_RECURSE cuda_helpers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/helper_includes/*.h")
  file(GLOB cuda_cufft_binaries "${CUDA_COMPILER_DIRECTORY}/cufft*.dll")
  
  set_source_files_properties(cuda_source_files PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(cuda_header_files PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(cuda_helpers PROPERTIES LANGUAGE CUDA)

  #remove all items contained in the cuda_helpers from the normal headers
  list(REMOVE_ITEM header_files cuda_helpers)

  #remove special files from the build
  list(REMOVE_ITEM cuda_source_files "src/cuenergy.cu")
  list(REMOVE_ITEM cuda_source_files "src/gpu_poisson_solver.cu")
  list(REMOVE_ITEM cuda_source_files "src/particles_kernel.cu")

  if(UNIX)
    list(REMOVE_ITEM source_files
      "src/dllmain.cpp"
      )
  endif()

  
  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files} ${cuda_header_files} ${cuda_source_files} ${cuda_helpers})
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE core geometry_calls protein_calls cufft)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  string(TOUPPER ${PROJECT_NAME} EXPORT_NAME)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  #install cufft (this seems to be necessary for every needed extra CUDA library)
  install(FILES ${cuda_cufft_binaries} DESTINATION "bin")

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("CUDA Helpers" FILES ${cuda_helpers})
  source_group("CUDA Header Files" FILES ${cuda_header_files})
  source_group("CUDA Source Files" FILES ${cuda_source_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Shaders" FILES ${shader_files})

endif(BUILD_PROTEIN_CUDA_PLUGIN)
