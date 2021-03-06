# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(OSPRay_plugin
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    geometry_calls
    mmstd_datatools
    protein_calls
    mesh)

if (OSPRay_plugin_PLUGIN_ENABLED)

  find_package(ospray CONFIG REQUIRED)
  find_package(rkcommon CONFIG REQUIRED)

  target_link_libraries(OSPRay_plugin PUBLIC ospray::ospray ospray::ospray_module_ispc rkcommon::rkcommon)

  if (WIN32)
    install(DIRECTORY "${OSPRAY_ROOT}/bin/" DESTINATION "bin" FILES_MATCHING PATTERN "*.dll")
  else ()
    if (TARGET ospray::ospray)
      get_target_property(OSPRAY_LIB_PATH ospray::ospray IMPORTED_LOCATION_RELEASE)
      get_filename_component(dir ${OSPRAY_LIB_PATH} DIRECTORY)
      install(DIRECTORY "${dir}" DESTINATION "${CMAKE_INSTALL_PREFIX}" FILES_MATCHING PATTERN "lib*")
    else (TARGET ospray::ospray)
      foreach (lib ${OSPRAY_LIBRARIES})
        get_filename_component(dir ${lib} DIRECTORY)
        install(DIRECTORY "${dir}" DESTINATION "${CMAKE_INSTALL_PREFIX}" FILES_MATCHING PATTERN "lib*")
      endforeach ()
    endif (TARGET ospray::ospray)
  endif ()
endif ()
