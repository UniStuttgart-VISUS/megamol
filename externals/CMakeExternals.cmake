# MegaMol
# Copyright (c) 2020, MegaMol Dev Team
# All rights reserved.
#

# Require git
find_package(Git REQUIRED)

# Clone external script
if (NOT EXISTS "${CMAKE_BINARY_DIR}/script-externals")
  message(STATUS "Downloading external scripts")
  execute_process(COMMAND
    ${GIT_EXECUTABLE} clone -b v2.6 https://github.com/UniStuttgart-VISUS/megamol-cmake-externals.git script-externals --depth 1
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
endif ()

# Include external script
include("${CMAKE_BINARY_DIR}/script-externals/cmake/External.cmake")

# Commonly needed for path setup
include(GNUInstallDirs)

#
# Centralized function to require externals to add them once by invoking
# require_external(<EXTERNAL_TARGET>).
#
# Think of this function as a big switch, testing for the name and presence
# of the external target to guard against duplicated targets.
#
function(require_external NAME)
  set(FETCHCONTENT_QUIET ON CACHE BOOL "")

  # ###########################################################################
  # ### Header-only libraries #################################################
  # ###########################################################################

  # glowl
  if (NAME STREQUAL "glowl")
    if (TARGET glowl)
      return()
    endif ()

    add_external_headeronly_project(glowl
      GIT_REPOSITORY https://github.com/invor/glowl.git
      GIT_TAG "dafee75f11c5d759df30ff651d6763e4e674dd0e"
      INCLUDE_DIR "include")
    target_compile_definitions(glowl INTERFACE GLOWL_OPENGL_INCLUDE_GLAD2)

  # ###########################################################################
  # ### Built libraries #######################################################
  # ###########################################################################

  # glad
  elseif (NAME STREQUAL "glad")
    if (TARGET glad)
      return()
    endif ()

    if (WIN32)
      set(GLAD_LIB "lib/glad.lib")
    else ()
      set(GLAD_LIB "${CMAKE_INSTALL_LIBDIR}/libglad.a")
    endif ()

    add_external_project(glad STATIC
      SOURCE_DIR glad
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${GLAD_LIB}")

    add_external_library(glad
      PROJECT glad
      LIBRARY ${GLAD_LIB})

  # IceT
  elseif (NAME STREQUAL "IceT")
    if (TARGET IceTCore)
      return()
    endif ()

    if (WIN32)
      set(ICET_CORE_LIB "lib/IceTCore.lib")
      set(ICET_GL_LIB "lib/IceTGL.lib")
      set(ICET_MPI_LIB "lib/IceTMPI.lib")
    else ()
      set(ICET_CORE_LIB "lib/libIceTCore.a")
      set(ICET_GL_LIB "lib/libIceTGL.a")
      set(ICET_MPI_LIB "lib/libIceTMPI.a")
    endif ()

    add_external_project(IceT STATIC
      GIT_REPOSITORY https://gitlab.kitware.com/icet/icet.git
      GIT_TAG abf5bf2b92c0531170c8db2621b375065c7da7c4 # master on 2021-07-26, because nothing was specified here.
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ICET_CORE_LIB}" "<INSTALL_DIR>/${ICET_GL_LIB}" "<INSTALL_DIR>/${ICET_MPI_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DICET_BUILD_TESTING=OFF
        -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

    add_external_library(IceTCore
      PROJECT IceT
      LIBRARY ${ICET_CORE_LIB})

    add_external_library(IceTGL
      PROJECT IceT
      LIBRARY ${ICET_GL_LIB})

    add_external_library(IceTMPI
      PROJECT IceT
      LIBRARY ${ICET_MPI_LIB})

  # libigl
  elseif (NAME STREQUAL "libigl")
    if (TARGET libigl)
      return()
    endif ()

    if (WIN32)
      set(LIBIGL_LIB "")
    else ()
      set(LIBIGL_LIB "")
    endif ()

    add_external_headeronly_project(libigl
      GIT_REPOSITORY https://github.com/libigl/libigl.git
      GIT_TAG "v2.1.0"
      INCLUDE_DIR "include")

  # obj-io
  elseif (NAME STREQUAL "obj-io")
    if (TARGET obj-io)
      return()
    endif ()

    add_external_headeronly_project(obj-io INTERFACE
      GIT_REPOSITORY https://github.com/thinks/obj-io.git
      GIT_TAG bfe835200fdff49b45a6de4561741203f85ad028 # master on 2021-07-26, because nothing was specified here.
      INCLUDE_DIR "include/thinks")

  # qhull
  elseif (NAME STREQUAL "qhull")
    if (TARGET qhull)
      return()
    endif ()

    if (WIN32)
      set(QHULL_LIB "lib/qhull<SUFFIX>.lib")
    else ()
      set(QUHULL_LIB "lib/libqhull<SUFFIX>.a")
    endif ()

    add_external_project(qhull STATIC
      GIT_REPOSITORY https://github.com/qhull/qhull.git
      GIT_TAG "v7.3.2"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${QHULL_LIB}"
      DEBUG_SUFFIX _d
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/qhull/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    add_external_library(qhull
      INCLUDE_DIR "include"
      LIBRARY ${QHULL_LIB}
      DEBUG_SUFFIX _d)

  # quickhull
  elseif (NAME STREQUAL "quickhull")
    if (TARGET quickhull)
      return()
    endif ()

    if (WIN32)
      set(QUICKHULL_LIB "lib/quickhull.lib")
    else ()
      set(QUICKHULL_LIB "lib/libquickhull.a")
    endif ()

    add_external_project(quickhull STATIC
      GIT_REPOSITORY https://github.com/akuukka/quickhull.git
      GIT_TAG 4f65e0801b8f60c9a97da2dadbe63c2b46397694 # master on 2021-07-26, because nothing was specified here.
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${QUICKHULL_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/quickhull/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    add_external_library(quickhull
      LIBRARY ${QUICKHULL_LIB})

  # snappy
  elseif (NAME STREQUAL "snappy")
    if (TARGET snappy)
      return()
    endif ()

    if (WIN32)
      set(SNAPPY_LIB "lib/snappy.lib")
    else ()
      set(SNAPPY_LIB "${CMAKE_INSTALL_LIBDIR}/libsnappy.a")
    endif ()

    add_external_project(snappy STATIC
      GIT_REPOSITORY https://github.com/google/snappy.git
      GIT_TAG "1.1.7"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${SNAPPY_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DSNAPPY_BUILD_TESTS=OFF
        -DCMAKE_BUILD_TYPE=Release)

    add_external_library(snappy
      LIBRARY ${SNAPPY_LIB})

  # tracking
  elseif (NAME STREQUAL "tracking")
    if (TARGET tracking)
      return()
    endif ()

    if (NOT WIN32)
      message(WARNING "External 'tracking' requested, but not available on non-Windows systems")
    endif ()

    set(TRACKING_LIB "lib/tracking.lib")
    set(TRACKING_NATNET_LIB "lib/NatNetLib.lib")

    add_external_project(tracking STATIC
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/mm-tracking.git
      GIT_TAG "v2.0"
      BUILD_BYPRODUCTS
        "<INSTALL_DIR>/${TRACKING_LIB}"
        "<INSTALL_DIR>/${TRACKING_NATNET_LIB}"
      CMAKE_ARGS
        -DCREATE_TRACKING_TEST_PROGRAM=OFF)

    add_external_library(tracking
      LIBRARY ${TRACKING_LIB})

    add_external_library(natnet
      PROJECT tracking
      LIBRARY ${TRACKING_NATNET_LIB})

    external_get_property(tracking SOURCE_DIR)
    set(tracking_files "${SOURCE_DIR}/tracking/conf/tracking.conf" PARENT_SCOPE)

  # zfp
  elseif (NAME STREQUAL "zfp")
    if (TARGET zfp)
      return()
    endif ()

    if (WIN32)
      set(ZFP_LIB "lib/zfp.lib")
    else ()
      set(ZFP_LIB "${CMAKE_INSTALL_LIBDIR}/libzfp.a")
    endif ()

    add_external_project(zfp STATIC
      GIT_REPOSITORY https://github.com/LLNL/zfp.git
      GIT_TAG "0.5.2"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZFP_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_UTILITIES=OFF
        -DBUILD_TESTING=OFF
        -DZFP_WITH_ALIGNED_ALLOC=ON
        -DZFP_WITH_CACHE_FAST_HASH=ON
        -DCMAKE_BUILD_TYPE=Release)

    add_external_library(zfp
      LIBRARY ${ZFP_LIB})

  # vr interop mwk-mint
  elseif(NAME STREQUAL "mwk-mint")
    if(TARGET mwk-mint)
      return()
    endif()

    if (MSVC_IDE)
      set(MSVC_TOOLSET "-${CMAKE_VS_PLATFORM_TOOLSET}")
    else ()
      set(MSVC_TOOLSET "")
    endif ()

    if(WIN32)
      set(MWKMint_LIB "${CMAKE_INSTALL_LIBDIR}/interop.lib")
      set(MWKMint_Spout_LIB "${CMAKE_INSTALL_LIBDIR}/Spout2.lib")
      set(MWKMint_ZMQ_LIB "${CMAKE_INSTALL_LIBDIR}/libzmq${MSVC_TOOLSET}-mt-sgd-4_3_5.lib")
    else()
      set(MWKMint_LIB "")
    endif()

    add_external_project(mwk-mint STATIC
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/MWK-mint/
      GIT_TAG "master"
      BUILD_BYPRODUCTS
        "<INSTALL_DIR>/${MWKMint_LIB}"
        "<INSTALL_DIR>/${MWKMint_Spout_LIB}"
        "<INSTALL_DIR>/${MWKMint_ZMQ_LIB}"
    )

    add_external_library(interop
      PROJECT mwk-mint
      LIBRARY ${MWKMint_LIB}
    )

    add_external_library(Spout2
      PROJECT mwk-mint
      LIBRARY ${MWKMint_Spout_LIB}
    )

  else ()
    message(FATAL_ERROR "Unknown external required \"${NAME}\"")
  endif ()

  mark_as_advanced(FORCE FETCHCONTENT_BASE_DIR)
  mark_as_advanced(FORCE FETCHCONTENT_FULLY_DISCONNECTED)
  mark_as_advanced(FORCE FETCHCONTENT_QUIET)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED)
endfunction(require_external)
