# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(moldyn_gl
  BUILD_DEFAULT ON
  DEPENDS_FEATURES
    #cuda
    opengl
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    moldyn
    datatools
    geometry_calls)

# if (moldyn_gl_PLUGIN_ENABLED)
# message(INFO " Have plugin")
# if(MEGAMOL_USE_CUDA)
# message(INFO " Have CUda")
# target_link_libraries(moldyn_gl
#       PRIVATE moldyn_cuda)
# endif()
# endif()

# if (moldyn_gl_PLUGIN_ENABLED)
#   get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
#   file(GLOB_RECURSE cuda_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cu")
#   set_source_files_properties(cuda_source_files PROPERTIES LANGUAGE CUDA)
#   target_sources(moldyn_gl PRIVATE ${cuda_source_files})
#   set_target_properties(moldyn_gl PROPERTIES CUDA_STANDARD 17)
#   target_include_directories(moldyn_gl PUBLIC "${CUDA_BIN_DIR}/../include")
#   source_group("CUDA Source Files" FILES ${cuda_source_files})
# endif()

if (moldyn_gl_PLUGIN_ENABLED)
if(MEGAMOL_USE_CUDA)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/")
  get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
  file(GLOB_RECURSE cuda_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "cuda/*.cu")
  set_source_files_properties(cuda_source_files PROPERTIES LANGUAGE CUDA)
  target_sources(moldyn_gl PRIVATE ${cuda_source_files})
  set_target_properties(moldyn_gl PROPERTIES CUDA_STANDARD 17)
  target_include_directories(moldyn_gl PUBLIC "${CUDA_BIN_DIR}/../include" "${CMAKE_CURRENT_SOURCE_DIR}/cuda/")
  source_group("CUDA Source Files" FILES ${cuda_source_files})
endif()
option(NVPERF "Enable Nsight Perf instrumentation" ON)
IF(${NVPERF})
    find_package(NvPerf REQUIRED)
    find_package(NvPerfUtility REQUIRED)
    target_link_libraries(moldyn_gl PRIVATE NvPerf-shared NvPerfUtility NvPerfUtilityImportsRyml)
    target_compile_definitions(moldyn_gl PRIVATE USE_NVPERF)
ENDIF()
endif()
