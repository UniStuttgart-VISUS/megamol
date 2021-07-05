# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(adios_plugin
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd_datatools)

if (adios_plugin_PLUGIN_ENABLED)
  option(USE_EXTERNAL_ADIOS "Option to use external ADIOS2" OFF)

  if (USE_EXTERNAL_ADIOS)
    find_package(ADIOS2 REQUIRED)
    target_link_libraries(adios_plugin PRIVATE adios2::adios2)

    include(GNUInstallDirs)
    install(DIRECTORY ${ADIOS2_DIR}/../../ DESTINATION ${CMAKE_INSTALL_LIBDIR})
  else ()
    require_external(adios2)
    target_link_libraries(adios_plugin PRIVATE adios2)
  endif ()

  if (MPI_C_FOUND)
    target_link_libraries(adios_plugin PRIVATE MPI::MPI_C)
  endif ()
endif ()
