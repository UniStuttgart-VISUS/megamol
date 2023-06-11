# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
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
    # plugin1
    # plugin2
  )

if (test_gl_PLUGIN_ENABLED)
  # Do additional plugin specific stuff, not covered by the generic plugin creation,
  # i.e. find_package() dependencies, additional sources 'target_sources(megamolplugin PRIVATE ...)', ....
  option(USE_NVPERF "Enable Nsight Perf instrumentation" OFF)
  if(${USE_NVPERF})
    find_package(NvPerf REQUIRED)
    find_package(NvPerfUtility REQUIRED)
    target_link_libraries(test_gl PRIVATE NvPerf-shared NvPerfUtility NvPerfUtilityImportsRyml)
    target_compile_definitions(test_gl PRIVATE USE_NVPERF)
  endif()
endif ()
