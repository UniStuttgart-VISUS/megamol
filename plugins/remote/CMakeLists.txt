# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(remote
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    cluster
    image_calls
  DEPENDS_EXTERNALS
    snappy
    zfp)

if (remote_PLUGIN_ENABLED)
  find_package(ZeroMQ CONFIG REQUIRED)
  find_package(cppzmq CONFIG REQUIRED)
  find_package(glm CONFIG REQUIRED)

  target_link_libraries(remote
    PRIVATE
      libzmq
      cppzmq
      glm::glm)

  if (MPI_C_FOUND)
    require_external(IceT)
    target_link_libraries(remote PRIVATE IceTCore IceTGL IceTMPI MPI::MPI_C)
  endif ()
endif ()
