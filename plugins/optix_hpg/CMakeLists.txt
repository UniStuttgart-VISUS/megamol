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

  find_package(CUDAToolkit)
  include(configure_optix)

  set(ptx_include_dirs "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/cuda_resources")
  embed_ptx(
    OUTPUT_TARGET
      embedded_raygen_programs
    PTX_LINK_LIBRARIES
      glm::glm
      OptiX::OptiX
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/raygen.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_miss_programs
    PTX_LINK_LIBRARIES
      glm::glm
      OptiX::OptiX
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/miss.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_miss_occlusion_programs
    PTX_LINK_LIBRARIES
      glm::glm
      OptiX::OptiX
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/miss_occlusion.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_sphere_programs
    PTX_LINK_LIBRARIES
      glm::glm
      OptiX::OptiX
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/sphere.cu)
  #embed_ptx(
  #  OUTPUT_TARGET
  #    embedded_sphere_occlusion_programs
  #  PTX_LINK_LIBRARIES
  #    glm::glm
  #    OptiX::OptiX
  #  PTX_INCLUDE_DIRECTORIES
  #    ${ptx_include_dirs}
  #  SOURCES
  #    cuda_resources/sphere_occlusion.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_mesh_programs
    PTX_LINK_LIBRARIES
      glm::glm
      OptiX::OptiX
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/mesh.cu)
  embed_ptx(
    OUTPUT_TARGET
      embedded_transitioncalculator_programs
    PTX_LINK_LIBRARIES
      glm::glm
      OptiX::OptiX
    PTX_INCLUDE_DIRECTORIES
      ${ptx_include_dirs}
    SOURCES
      cuda_resources/transitioncalculator.cu)

  target_link_libraries(optix_hpg PRIVATE
    $<TARGET_OBJECTS:embedded_raygen_programs>
    $<TARGET_OBJECTS:embedded_miss_programs>
    $<TARGET_OBJECTS:embedded_miss_occlusion_programs>
    $<TARGET_OBJECTS:embedded_sphere_programs>
    #$<TARGET_OBJECTS:embedded_sphere_occlusion_programs>
    $<TARGET_OBJECTS:embedded_mesh_programs>
    $<TARGET_OBJECTS:embedded_transitioncalculator_programs>)

  target_include_directories(optix_hpg PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda_resources/")
  target_link_libraries(optix_hpg PRIVATE CUDA::cuda_driver OptiX::OptiX)
  if (MEGAMOL_USE_OPENGL)
    target_link_libraries(optix_hpg PRIVATE mmstd_gl)
  endif ()
endif ()
