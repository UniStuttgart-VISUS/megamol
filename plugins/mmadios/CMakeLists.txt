# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmadios
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    datatools
    geometry_calls)

if (mmadios_PLUGIN_ENABLED)
  option(USE_EXTERNAL_ADIOS "Option to use external ADIOS2" OFF)

  if (USE_EXTERNAL_ADIOS)
    find_package(ADIOS2 REQUIRED)
    target_link_libraries(mmadios PRIVATE adios2::adios2)

    include(GNUInstallDirs)
    install(DIRECTORY ${ADIOS2_DIR}/../../ DESTINATION ${CMAKE_INSTALL_LIBDIR})
  else ()
    require_external(adios2)
    target_link_libraries(mmadios PRIVATE adios2)
  endif ()

  if (MPI_C_FOUND)
    target_link_libraries(mmadios PRIVATE MPI::MPI_C)
  endif ()
endif ()
