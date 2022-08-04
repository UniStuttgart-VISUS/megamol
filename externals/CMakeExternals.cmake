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
  # ### Built libraries #######################################################
  # ###########################################################################

  # IceT
  if (NAME STREQUAL "IceT")
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
