# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

find_package(ZeroMQ REQUIRED)
find_package(cppzmq REQUIRED)

megamol_plugin(remote
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    cluster
    image_calls
  DEPENDS_LIBRARIES
    libzmq
    cppzmq
  DEPENDS_EXTERNALS
    glm
    snappy
    zfp)

if (remote_PLUGIN_ENABLED)
  if (MPI_C_FOUND)
    require_external(IceT)
    target_link_libraries(remote PRIVATE IceTCore IceTGL IceTMPI MPI::MPI_C)
  endif ()
endif ()
