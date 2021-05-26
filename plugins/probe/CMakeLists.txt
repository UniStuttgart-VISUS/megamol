# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(probe
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    adios_plugin
    mesh
  DEPENDS_EXTERNALS
    Eigen
    nanoflann
    libigl
    asmjit
    blend2d
    qhull)

if (probe_PLUGIN_ENABLED)
  file(GLOB_RECURSE third RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/*")
  file(GLOB_RECURSE ospray_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "probe_ospray/src/*.cpp")
  file(GLOB_RECURSE ospray_headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "probe_ospray/src/*.h")
  file(GLOB_RECURSE ospray_public_headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "probe_ospray/include/*.h")

  target_sources(probe PRIVATE ${third})
  target_include_directories(probe PUBLIC "3rd")

  if (BUILD_OSPRAY_PLUGIN_PLUGIN)
    target_sources(probe PRIVATE ${ospray_sources} ${ospray_headers} ${ospray_public_headers})
    target_compile_definitions(probe PRIVATE ${EXPORT_NAME}_HAS_OSPRAY)
    target_include_directories(probe PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/probe_ospray/include> "probe_ospray/src")
    target_link_libraries(probe PUBLIC OSPRay_plugin)
  endif ()

  source_group("3rd Files" FILES ${third})
endif ()
