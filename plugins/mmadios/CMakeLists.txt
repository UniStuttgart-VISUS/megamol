# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmadios
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mmstd
    cluster
    datatools
    geometry_calls)

if (mmadios_PLUGIN_ENABLED)
  find_package(ADIOS2 REQUIRED)
  target_link_libraries(mmadios PRIVATE adios2::adios2)

  if (MPI_C_FOUND)
    target_link_libraries(mmadios PRIVATE MPI::MPI_C)
  endif ()
endif ()
