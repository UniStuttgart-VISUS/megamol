# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmospray
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmadios
    mmstd
    geometry_calls
    datatools
    protein_calls
    mesh)

if (mmospray_PLUGIN_ENABLED)

  find_package(ospray CONFIG REQUIRED)
  find_package(rkcommon CONFIG REQUIRED)

  target_link_libraries(mmospray PUBLIC ospray::ospray rkcommon::rkcommon)

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
