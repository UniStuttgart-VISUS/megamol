# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(remote
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    cluster
    image_calls
  DEPENDS_EXTERNALS
    zfp)

if (remote_PLUGIN_ENABLED)
  find_package(ZeroMQ CONFIG REQUIRED)
  find_package(cppzmq CONFIG REQUIRED)
  find_package(glm CONFIG REQUIRED)
  find_package(snappy CONFIG REQUIRED)

  target_link_libraries(remote
    PRIVATE
      libzmq
      cppzmq
      glm::glm)

  if (MPI_C_FOUND)
    require_external(IceT)
    target_link_libraries(remote PRIVATE IceTCore IceTGL IceTMPI MPI::MPI_C vislib_gl)
  endif ()
endif ()
