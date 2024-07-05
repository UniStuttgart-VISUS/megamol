# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(optix_hpg
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    cuda
  DEPENDS_PLUGINS
    mmstd
    mesh
    geometry_calls)

if (optix_hpg_PLUGIN_ENABLED)
  find_package(glm CONFIG REQUIRED)
  target_link_libraries(optix_hpg
    PRIVATE
      glm::glm)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

  set(OPTIX_HPG_CUDA_ARCH "75" CACHE STRING "CUDA Arch for Optix HPG")
  set_property(CACHE OPTIX_HPG_CUDA_ARCH PROPERTY STRINGS 61 75 80)

  if (OPTIX_HPG_CUDA_ARCH STREQUAL "61")
    set(CMAKE_CUDA_ARCHITECTURES 61)
  elseif (OPTIX_HPG_CUDA_ARCH STREQUAL "75")
    set(CMAKE_CUDA_ARCHITECTURES 75)
  elseif (OPTIX_HPG_CUDA_ARCH STREQUAL "80")
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif ()

  find_package(CUDAToolkit)
  include(configure_optix)

  #get_target_property(GLM_INCLUDES glm::glm INTERFACE_INCLUDE_DIRECTORIES)

  # Collect source files
  file(GLOB_RECURSE cuda_resource_include_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "cuda_resources/*.h" "cuda_resources/*.cuh")

  #include_directories(${GLM_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/cuda_resources")
  set(ptx_include_dirs "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/cuda_resources" "${OptiX_INCLUDE}")
  embed_ptx(
    OUTPUT_TARGET
      embedded_raygen_programs
    PTX_LINK_LIBRARIES
      glm::glm
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/raygen.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_miss_programs
    PTX_LINK_LIBRARIES
      glm::glm
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/miss.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_miss_occlusion_programs
    PTX_LINK_LIBRARIES
      glm::glm
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/miss_occlusion.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_sphere_programs
    PTX_LINK_LIBRARIES
      glm::glm
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/sphere.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_sphere_occlusion_programs
    PTX_LINK_LIBRARIES
      glm::glm
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/sphere_occlusion.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_mesh_programs
    PTX_LINK_LIBRARIES
      glm::glm
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/mesh.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_transitioncalculator_programs
    PTX_LINK_LIBRARIES
      glm::glm
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/transitioncalculator.cu)

  target_sources(optix_hpg PRIVATE ${cuda_resource_include_files})
  #target_sources(optix_hpg PRIVATE ${cuda_resource_include_files} ${embedded_raygen_programs} ${embedded_miss_programs} ${embedded_miss_occlusion_programs} ${embedded_sphere_programs} ${embedded_sphere_occlusion_programs} ${embedded_mesh_programs} ${embedded_transitioncalculator_programs})
  target_link_libraries(optix_hpg PRIVATE $<TARGET_OBJECTS:embedded_raygen_programs> $<TARGET_OBJECTS:embedded_miss_programs> $<TARGET_OBJECTS:embedded_miss_occlusion_programs> $<TARGET_OBJECTS:embedded_sphere_programs> $<TARGET_OBJECTS:embedded_sphere_occlusion_programs> $<TARGET_OBJECTS:embedded_mesh_programs> $<TARGET_OBJECTS:embedded_transitioncalculator_programs>)

  target_include_directories(optix_hpg PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda_resources/" "${OptiX_INCLUDE}")
  target_link_libraries(optix_hpg PRIVATE CUDA::cuda_driver)
  if (MEGAMOL_USE_OPENGL)
    target_link_libraries(optix_hpg PRIVATE mmstd_gl)
  endif ()
endif ()
