#
# MegaMol™ molecularmaps Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER molecularmaps EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build molecularmaps" OFF)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  if(NOT ENABLE_CUDA)
    message(FATAL_ERROR "The molecularmaps plugin requires CUDA. Please enable it setting the option ENABLE_CUDA.")
  endif()

  project(molecularmaps LANGUAGES CXX CUDA)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_PROTEIN_CALLS_PLUGIN BUILD_GEOMETRY_CALLS_PLUGIN" CACHE INTERNAL "")

  if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 7)
    message(FATAL_ERROR "Molecularmaps currently only supports CUDA Version 8 or above. The version found was CUDA ${CUDA_VERSION_MAJOR} (${CMAKE_CUDA_COMPILER_VERSION}). Please turn off the Molecularmaps plugin or install a correct version of CUDA." )
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

  get_filename_component(cuda_compiler_directory "${CMAKE_CUDA_COMPILER}" DIRECTORY)

  # Add externals.
  require_external(Eigen)

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")
  file(GLOB_RECURSE cuda_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cu")
  file(GLOB_RECURSE cuda_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cuh")
  file(GLOB_RECURSE cuda_helpers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/helper_includes/*.h")

  # Set language to CUDA for respective files.
  set_source_files_properties(cuda_source_files PROPERTIES LANGUAGE CUDA)

  if(UNIX)
    list(REMOVE_ITEM source_files "src/dllmain.cpp")
  endif()

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files} ${cuda_header_files} ${cuda_source_files})
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg" CUDA_STANDARD 14)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src" "src/helper_includes")
  target_link_libraries(${PROJECT_NAME} PRIVATE vislib core glad protein_calls geometry_calls Eigen)

  if(NOT MSVC)
    get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
    target_include_directories(${PROJECT_NAME} PUBLIC "${CUDA_BIN_DIR}/../include")
    target_compile_definitions(${PROJECT_NAME} PRIVATE __CUDACC__)
  endif()

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
  source_group("Source Files" FILES ${source_files})
  source_group("CUDA Header Files" FILES ${cuda_header_files})
  source_group("CUDA Source Files" FILES ${cuda_source_files})
  source_group("Shaders" FILES ${shader_files})

endif(BUILD_${EXPORT_NAME}_PLUGIN)
