#
# MegaMol™ protein_cuda Plugin
# Copyright 2015, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_PROTEIN_CUDA_PLUGIN "Option to build protein_cuda" OFF)

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(BUILD_PROTEIN_CUDA_PLUGIN)
  project(protein_cuda LANGUAGES CXX CUDA)

  #NVCC options
  if(NOT MSVC)
    if(NOT "--std" IN_LIST CUDA_NVCC_FLAGS)
      list(APPEND CUDA_NVCC_FLAGS "--std c++11")
    endif()
  endif()
  list(APPEND CUDA_NVCC_FLAGS "-03")
  list(APPEND CUDA_NVCC_FLAGS "-gencode")
  list(APPEND CUDA_NVCC_FLAGS "-arch=compute_30, code=sm_30")
  
  set(LIBS vislib core mmstd_trisoup protein_calls glload)
  include_directories("include" "src" "src/helper_includes")


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
  
  set_source_files_properties(cuda_source_files PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(cuda_header_files PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(cuda_helpers PROPERTIES LANGUAGE CUDA)

  #remove all items contained in the cuda_helpers from the normal headers
  list(REMOVE_ITEM header_files cuda_helpers)

  if(UNIX)
    list(REMOVE_ITEM source_files
      "src/dllmain.cpp"
      )
  endif()

  
  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files} ${cuda_header_files} ${cuda_source_files} ${cuda_helpers})
  add_dependencies(${PROJECT_NAME} vislib glload core mmstd_trisoup protein_calls)
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
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

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("CUDA Helpers" FILES ${cuda_helpers})
  source_group("CUDA Header Files" FILES ${cuda_header_files})
  source_group("CUDA Source Files" FILES ${cuda_source_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Shaders" FILES ${shader_files})

endif(BUILD_PROTEIN_CUDA_PLUGIN)
