#
# MegaMol optix_hpg Plugin
# Copyright 2021, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER optix_hpg EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build optix_hpg" OFF)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  if(NOT ENABLE_CUDA)
    message(FATAL_ERROR "The OPTIX_HPG plugin requires CUDA. Please enable it setting the option ENABLE_CUDA.")
  endif()
  project(optix_hpg LANGUAGES CXX CUDA)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

  set(OPTIX_HPG_CUDA_ARCH "75" CACHE STRING "CUDA Arch for Optix HPG")
  set_property(CACHE OPTIX_HPG_CUDA_ARCH PROPERTY STRINGS 61 75 80)

  if (OPTIX_HPG_CUDA_ARCH STREQUAL "61")
    set(CMAKE_CUDA_ARCHITECTURES 61)
  elseif (OPTIX_HPG_CUDA_ARCH STREQUAL "75")
    set(CMAKE_CUDA_ARCHITECTURES 75)
  elseif (OPTIX_HPG_CUDA_ARCH STREQUAL "80")
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()

  include(configure_optix)
  find_package(CUDAToolkit)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_MESH_PLUGIN" CACHE INTERNAL "")

  require_external(glm)
  get_target_property(GLM_INCLUDES glm INTERFACE_INCLUDE_DIRECTORIES)

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  #file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")
  file(GLOB_RECURSE cuda_resource_include_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "cuda_resources/*.h")

  include_directories(${GLM_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/src")
  cuda_compile_and_embed(embedded_raygen_programs cuda_resources/raygen.cu)
  cuda_compile_and_embed(embedded_miss_programs  cuda_resources/miss.cu)
  cuda_compile_and_embed(embedded_sphere_programs cuda_resources/sphere.cu)
  cuda_compile_and_embed(embedded_mesh_programs cuda_resources/mesh.cu)
  cuda_compile_and_embed(embedded_transitioncalculator_programs cuda_resources/transitioncalculator.cu)

  # Target definition
  #add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files} ${cuda_resource_include_files}
    ${embedded_raygen_programs} ${embedded_miss_programs} ${embedded_sphere_programs} ${embedded_mesh_programs} ${embedded_transitioncalculator_programs})
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    "include" "src" "${CMAKE_CURRENT_SOURCE_DIR}/cuda_resources/")
  target_link_libraries(${PROJECT_NAME} PRIVATE core glm CUDA::cuda_driver mesh)

  # Installation rules for generated files
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    #install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    #install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    #install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  #source_group("Shaders" FILES ${shader_files})

  # Register plugin
  megamol_register_plugin(${PROJECT_NAME})
endif()
