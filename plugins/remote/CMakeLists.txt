# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(remote
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    image_calls
  DEPENDS_EXTERNALS
    glm
    libzmq
    libcppzmq
    snappy
    zfp)

if (remote_PLUGIN_ENABLED)
  if (MPI_C_FOUND)
    require_external(IceT)
    target_link_libraries(remote PRIVATE IceTCore IceTGL IceTMPI MPI::MPI_C)
  endif ()
endif ()
