include(External)
include(FetchContent)

#
# Centralized function to require externals to add them once by invoking
# require_external(<EXTERNAL_TARGET>).
#
# Think of this function as a big switch, testing for the name and presence 
# of the external target to guard against duplicated targets.
#
function(require_external NAME)
  set(FETCHCONTENT_QUIET ON CACHE BOOL "" FORCE)
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)

  if(NAME STREQUAL "libzmq" OR NAME STREQUAL "libcppzmq")
    if(TARGET libzmq OR TARGET libcppzmq)
      return()
    endif()

    set(ZMQ_VER "4_3_3")
    string(REPLACE "_" "." ZMQ_TAG "v${ZMQ_VER}")

    if(MSVC_IDE)
      set(MSVC_TOOLSET "-${CMAKE_VS_PLATFORM_TOOLSET}")
    else()
      set(MSVC_TOOLSET "")
    endif()

    if(WIN32)
      set(ZMQ_IMPORT_DEBUG "lib/libzmq${MSVC_TOOLSET}-mt-gd-${ZMQ_VER}.lib")
      set(ZMQ_IMPORT_RELEASE "lib/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.lib")
      set(ZMQ_DEBUG "bin/libzmq${MSVC_TOOLSET}-mt-gd-${ZMQ_VER}.dll")
      set(ZMQ_RELEASE "bin/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.dll")
    else()
      include(GNUInstallDirs)
      set(ZMQ_IMPORT_DEBUG "")
      set(ZMQ_IMPORT_RELEASE "")
      set(ZMQ_DEBUG "${CMAKE_INSTALL_LIBDIR}/libzmq.so")
      set(ZMQ_RELEASE ${ZMQ_DEBUG})
    endif()

    add_external_project(libzmq_ext
      GIT_REPOSITORY https://github.com/zeromq/libzmq.git
      GIT_TAG 56ace6d03f521b9abb5a50176ec7763c1b77afa9 # We need https://github.com/zeromq/libzmq/pull/3636
      #GIT_TAG ${ZMQ_TAG}
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZMQ_IMPORT_DEBUG}" "<INSTALL_DIR>/${ZMQ_IMPORT_RELEASE}"
      CMAKE_ARGS
        -DZMQ_BUILD_TESTS=OFF
        -DENABLE_PRECOMPILED=OFF)

      add_external_library(libzmq SHARED
      DEPENDS libzmq_ext
      IMPORT_LIBRARY_DEBUG ${ZMQ_IMPORT_DEBUG}
      IMPORT_LIBRARY_RELEASE ${ZMQ_IMPORT_RELEASE}
      LIBRARY_DEBUG ${ZMQ_DEBUG}
      LIBRARY_RELEASE ${ZMQ_RELEASE})

    add_external_project(libcppzmq_ext
      DEPENDS libzmq
      GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
      GIT_TAG "v4.4.1"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND "")

    add_external_library(libcppzmq INTERFACE
      DEPENDS libcppzmq_ext
      INCLUDE_DIR "src/libcppzmq_ext/")

  elseif(NAME STREQUAL "zlib")
    fetch_external(zlib zlib
      "CMAKE_POSITION_INDEPENDENT_CODE"
      ""
      "AMD64;ASM686;CMAKE_BACKWARDS_COMPATIBILITY;EXECUTABLE_OUTPUT_PATH;INSTALL_BIN_DIR;INSTALL_INC_DIR;INSTALL_LIB_DIR;INSTALL_MAN_DIR;INSTAL_PKGCONFIG_DIR;LIBRARY_OUTPUT_PATH"
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG "v1.2.11")

  elseif(NAME STREQUAL "libpng")
    if(TARGET libpng)
      return()
    endif()

    require_external(zlib)

    if(MSVC)
      set(LIBPNG_DEBUG "lib/libpng16_staticd${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(LIBPNG_RELEASE "lib/libpng16_static${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
      include(GNUInstallDirs)
      set(LIBPNG_DEBUG "${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}png16${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(LIBPNG_RELEASE "${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}png16${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()

    get_target_property(INSTALL_DIR zlib INSTALL_DIR)

    add_external_project(libpng_ext
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/libpng.git
      GIT_TAG "v1.6.34"
      DEPENDS zlib
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${LIBPNG_DEBUG}" "<INSTALL_DIR>/${LIBPNG_RELEASE}"
      CMAKE_ARGS
        -DPNG_SHARED=OFF
        -DPNG_TESTS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_PREFIX_PATH:PATH=${INSTALL_DIR})

    add_external_library(libpng STATIC
      DEPENDS libpng_ext
      INCLUDE_DIR "include"
      LIBRARY_DEBUG ${LIBPNG_DEBUG}
      LIBRARY_RELEASE ${LIBPNG_RELEASE}
      INTERFACE_LIBRARIES zlib)

  elseif(NAME STREQUAL "zfp")
    fetch_external(zfp zfp
      "ZFP_WITH_ALIGNED_ALLOC;ZFP_WITH_CACHE_FAST_HASH"
      "BUILD_SHARED_LIBS;BUILD_UTILITIES;BUILD_TESTING"
      "BUILD_EXAMPLES;ZFP_BIT_STREAM_WORD_SIZE;ZFP_ENABLE_PIC;ZFP_WITH_BIT_STREAM_STRIDED;ZFP_WITH_CACHE_PROFILE;ZFP_WITH_CACHE_TWOWAY"
      GIT_REPOSITORY https://github.com/LLNL/zfp.git
      GIT_TAG "0.5.2")

  elseif(NAME STREQUAL "glm")
    fetch_external_headeronly(glm
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG "0.9.8")

  elseif(NAME STREQUAL "glowl")
    fetch_external_headeronly(glowl
      GIT_REPOSITORY https://github.com/invor/glowl.git
      GIT_TAG "v0.1")

  elseif(NAME STREQUAL "json")
    fetch_external_headeronly(json
      GIT_REPOSITORY https://github.com/azadkuh/nlohmann_json_release.git
      GIT_TAG "v3.5.0")

  elseif(NAME STREQUAL "Eigen")
    fetch_external_headeronly(Eigen
      GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
      GIT_TAG "3.3.4")

  elseif(NAME STREQUAL "nanoflann")
    fetch_external_headeronly(nanoflann
      GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
      GIT_TAG "v1.3.0")

  elseif(NAME STREQUAL "Delaunator")
    fetch_external_headeronly(Delaunator
      GIT_REPOSITORY https://github.com/delfrrr/delaunator-cpp.git
      GIT_TAG "v0.4.0")

  elseif(NAME STREQUAL "tracking")
    if(TARGET tracking)
      return()
    endif()

    set(TRACKING_LIB "bin/tracking.dll")
    set(TRACKING_IMPORT_LIB "lib/tracking.lib")
    set(TRACKING_NATNET_LIB "src/tracking_ext/tracking/natnet/lib/x64/NatNetLib.dll")
    set(TRACKING_NATNET_IMPORT_LIB "src/tracking_ext/tracking/natnet/lib/x64/NatNetLib.lib")

    add_external_project(tracking_ext
        GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/mm-tracking.git
         BUILD_BYPRODUCTS "<INSTALL_DIR>/${TRACKING_IMPORT_LIB}" "<INSTALL_DIR>/${TRACKING_NATNET_IMPORT_LIB}"
        CMAKE_ARGS 
          -DCREATE_TRACKING_TEST_PROGRAM=OFF)

    add_external_library(tracking SHARED 
        DEPENDS tracking_ext 
        IMPORT_LIBRARY_DEBUG ${TRACKING_IMPORT_LIB}
        IMPORT_LIBRARY_RELEASE ${TRACKING_IMPORT_LIB}
        LIBRARY_DEBUG ${TRACKING_LIB}
        LIBRARY_RELEASE ${TRACKING_LIB})

    add_external_library(natnet SHARED 
        DEPENDS tracking_ext 
        IMPORT_LIBRARY_DEBUG ${TRACKING_NATNET_IMPORT_LIB}
        IMPORT_LIBRARY_RELEASE ${TRACKING_NATNET_IMPORT_LIB}
        LIBRARY_DEBUG ${TRACKING_NATNET_LIB}     
        LIBRARY_RELEASE ${TRACKING_NATNET_LIB})

    add_external_library(tracking_int INTERFACE
      DEPENDS tracking_ext
      INCLUDE_DIR "src/tracking_ext/tracking/include")

  elseif(NAME STREQUAL "sim_sort")
    fetch_external_headeronly(sim_sort
      GIT_REPOSITORY https://github.com/alexstraub1990/simultaneous-sort.git)

  else()
    message(FATAL_ERROR "Unknown external required \"${NAME}\"")
  endif()

  # Hide fetch-content variables
  mark_as_advanced(FORCE FETCHCONTENT_BASE_DIR)
  mark_as_advanced(FORCE FETCHCONTENT_FULLY_DISCONNECTED)
  mark_as_advanced(FORCE FETCHCONTENT_QUIET)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED)
endfunction(require_external)
