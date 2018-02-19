#
# MegaMol™ volume_cuda Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER volume_cuda EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build volume_cuda" OFF)

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(volume_cuda LANGUAGES CXX CUDA)

  #NVCC options
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

  get_filename_component(cuda_compiler_directory "${CMAKE_CUDA_COMPILER}" DIRECTORY)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_PROTEIN_CALLS_PLUGIN BUILD_MMSTD_TRISOUP_PLUGIN" CACHE INTERNAL "")

  set(LIBS vislib core glad protein_calls mmstd_trisoup)
  if(MSVC)
    include_directories("include" "src" "src/helper_includes")
  else()
    get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
    include_directories("include" "src" "src/helper_includes" ${CUDA_BIN_DIR}/../include)
    add_definitions(-D__CUDACC__)
  endif()

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  #file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")
  # CUDA related files
  file(GLOB_RECURSE cuda_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cu")
  file(GLOB_RECURSE cuda_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cuh")
  file(GLOB_RECURSE cuda_helpers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/helper_includes/*.h")
  #file(GLOB_RECURSE cuda_std_header_files RELATIVE "${cuda_compiler_directory}/../include/*.h")

  set_source_files_properties(cuda_source_files PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(cuda_header_files PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(cuda_helpers PROPERTIES LANGUAGE CUDA)

  list(REMOVE_ITEM header_files cuda_helpers)

  if(UNIX)
    list(REMOVE_ITEM source_files "src/dllmain.cpp")
  endif()

  # Target definition
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files} ${cuda_header_files} ${cuda_source_files} ${cuda_helpers})
  add_dependencies(${PROJECT_NAME} vislib core protein_calls mmstd_trisoup)
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("CUDA Helpers" FILES ${cuda_helpers})
  source_group("CUDA Header Files" FILES ${cuda_header_files})
  source_group("CUDA Source Files" FILES ${cuda_source_files})
  source_group("Source Files" FILES ${source_files})
  #source_group("Shaders" FILES ${shader_files})



endif(BUILD_${EXPORT_NAME}_PLUGIN)
