# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(optix_owl
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    # feature1
    cuda
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    geometry_calls
    # plugin1
    # plugin2
  )

if (optix_owl_PLUGIN_ENABLED)
  # Do additional plugin specific stuff, not covered by the generic plugin creation,
  # i.e. find_package() dependencies, additional sources 'target_sources(megamolplugin PRIVATE ...)', ....
  find_package(CUDAToolkit)

  set(MM_OWL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../owl CACHE PATH "owl dir")
  # set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MM_OWL_DIR}/owl/common/cmake/")

  add_subdirectory(${MM_OWL_DIR} ${CMAKE_CURRENT_BINARY_DIR}/owl EXCLUDE_FROM_ALL)

  target_link_libraries(optix_owl PRIVATE owl::owl)

  embed_ptx(
    OUTPUT_TARGET
      raygen-ptx
    PTX_LINK_LIBRARIES
      owl::owl
    SOURCES
      cuda_resources/raygenPrograms.cu
  )
  embed_ptx(
    OUTPUT_TARGET
      pkd-ptx
    PTX_LINK_LIBRARIES
      owl::owl
    SOURCES
      cuda_resources/pkdPrograms.cu
  )
  embed_ptx(
    OUTPUT_TARGET
      treelets-ptx
    PTX_LINK_LIBRARIES
      owl::owl
    SOURCES
      cuda_resources/treeletsPrograms.cu
  )
  embed_ptx(
    OUTPUT_TARGET
      bvh-ptx
    PTX_LINK_LIBRARIES
      owl::owl
    SOURCES
      cuda_resources/bvhPrograms.cu
  )

  target_link_libraries(optix_owl PRIVATE
    $<TARGET_OBJECTS:raygen-ptx>
    $<TARGET_OBJECTS:pkd-ptx>
    $<TARGET_OBJECTS:treelets-ptx>
    $<TARGET_OBJECTS:bvh-ptx>)
  target_include_directories(optix_owl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cuda_resources")

endif ()
