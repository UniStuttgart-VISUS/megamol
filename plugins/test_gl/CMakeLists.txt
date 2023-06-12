# MegaMol
# Copyright (c) 2023, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(test_gl
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    geometry_calls
  )

if (test_gl_PLUGIN_ENABLED)
  option(USE_NVPERF "Enable Nsight Perf instrumentation" OFF)
  if(${USE_NVPERF})
    find_package(NvPerf REQUIRED)
    find_package(NvPerfUtility REQUIRED)
    target_link_libraries(test_gl PRIVATE NvPerf-shared NvPerfUtility NvPerfUtilityImportsRyml)
    target_compile_definitions(test_gl PRIVATE USE_NVPERF)
  endif()
endif ()
