#
# Centralized function to require externals to add them once by invoking
# require_external(<EXTERNAL_TARGET>).
#
# Think of this function as a big switch, testing for the name and presence 
# of the external target to guard against duplicated targets.
#
function(require_external NAME)
  if(NAME STREQUAL "libzmq" OR NAME STREQUAL "libcppzmq")
    if(TARGET libzmq OR TARGET libcppzmq)
      return()
    endif()
    
    set(ZMQ_VER "4_2_3")
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
      GIT_TAG ${ZMQ_TAG}
      CMAKE_ARGS
        -DZMQ_BUILD_TESTS=OFF)
      add_external_library(libzmq SHARED
      DEPENDS libzmq_ext
      IMPORT_LIBRARY_DEBUG ${ZMQ_IMPORT_DEBUG}
      IMPORT_LIBRARY_RELEASE ${ZMQ_IMPORT_RELEASE}
      LIBRARY_DEBUG ${ZMQ_DEBUG}
      LIBRARY_RELEASE ${ZMQ_RELEASE})

    add_external_project(libcppzmq_ext
      DEPENDS libzmq
      GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
      GIT_TAG ${ZMQ_TAG}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND "")
    add_external_library(libcppzmq INTERFACE
      DEPENDS libcppzmq_ext
      INCLUDE_DIR "src/libcppzmq_ext/")

  elseif(NAME STREQUAL "zlib")
    if(TARGET zlib)
      return()
    endif()

    if(MSVC)
      set(ZLIB_DEBUG "lib/zlibstaticd${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(ZLIB_RELEASE "lib/zlibstatic${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
      include(GNUInstallDirs)
      #set(ZLIB_DEBUG "${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
      #set(ZLIB_RELEASE "${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(ZLIB_DEBUG "lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(ZLIB_RELEASE "lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()
    add_external_project(zlib_ext
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG "v1.2.11"
      CMAKE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON)
    add_external_library(zlib STATIC
      DEPENDS zlib_ext
      INCLUDE_DIR "include"
      LIBRARY_DEBUG ${ZLIB_DEBUG}
      LIBRARY_RELEASE ${ZLIB_RELEASE})

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
    ExternalProject_Get_Property(zlib_ext INSTALL_DIR)
    add_external_project(libpng_ext
      GIT_REPOSITORY https://git.code.sf.net/p/libpng/code
      GIT_TAG "v1.6.34"
      DEPENDS zlib_ext
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
    if(TARGET zfp)
      return()
    endif()
    
    if(WIN32)
      set(ZFP_LIB "lib/zfp.lib")
    else()
      include(GNUInstallDirs)
      set(ZFP_LIB "${CMAKE_INSTALL_LIBDIR}/libzfp.a")
    endif()

    add_external_project(zfp_ext
      GIT_REPOSITORY https://github.com/LLNL/zfp.git
      GIT_TAG "0.5.2"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_UTILITIES=OFF
        -DBUILD_TESTING=OFF
        -DZFP_WITH_ALIGNED_ALLOC=ON
        -DZFP_WITH_CACHE_FAST_HASH=ON
        -DCMAKE_BUILD_TYPE=Release)
    add_external_library(zfp STATIC
      DEPENDS zfp_ext
      LIBRARY ${ZFP_LIB})

  elseif(NAME STREQUAL "glm")
    if(TARGET glm)
      return()
    endif()
    
    add_external_project(glm_ext
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG "0.9.8"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND ""
      CMAKE_ARGS -DGLM_TEST_ENABLE=OFF)
    add_external_library(glm INTERFACE
      DEPENDS glm_ext
      INCLUDE_DIR "src/glm_ext/")

  elseif(NAME STREQUAL "glowl")
    if(TARGET glowl)
      return()
    endif()
    
    add_external_project(glowl_ext
      GIT_REPOSITORY https://github.com/invor/glowl.git
      GIT_TAG "v0.1"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND "")
    add_external_library(glowl INTERFACE
      DEPENDS glowl_ext
      INCLUDE_DIR "src/glowl_ext/include")

  elseif(NAME STREQUAL "json")
    if(TARGET json)
      return()
    endif()
    
    add_external_project(json_ext
      GIT_REPOSITORY https://github.com/azadkuh/nlohmann_json_release.git
      GIT_TAG "v3.5.0"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND ""
      CMAKE_ARGS -DBUILD_TESTING=OFF)
    add_external_library(json INTERFACE
      DEPENDS json_ext
      INCLUDE_DIR "src/json_ext/")

  elseif(NAME STREQUAL "Eigen")
    if(TARGET Eigen)
      return()
    endif()
    
    add_external_project(Eigen_ext
      GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
      GIT_TAG "3.3.4"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND "")
    add_external_library(Eigen INTERFACE
      DEPENDS Eigen_ext
      INCLUDE_DIR "src/Eigen_ext")
	  
  elseif(NAME STREQUAL "nanoflann")
    if(TARGET nanoflann)
      return()
    endif()
    
    add_external_project(nanoflann_ext
      GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
      GIT_TAG "v1.3.0"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND "")
    add_external_library(nanoflann INTERFACE
      DEPENDS nanoflann_ext
      INCLUDE_DIR "src/nanoflann_ext/include")
	  
  elseif(NAME STREQUAL "Delaunator")
    if(TARGET Delaunator)
      return()
    endif()
	
    add_external_project(Delaunator_ext
      GIT_REPOSITORY https://github.com/delfrrr/delaunator-cpp.git
      GIT_TAG "v0.4.0"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND "")
    add_external_library(Delaunator INTERFACE
      DEPENDS Delaunator_ext
      INCLUDE_DIR "src/Delaunator_ext/include")

  elseif(NAME STREQUAL "tracking")
    if(TARGET tracking)
      return()
    endif()

    if(WIN32)
      set(TRACKING_LIB "bin/tracking.dll")
      set(TRACKING_IMPORT_LIB "lib/tracking.lib")
      set(TRACKING_NATNET_LIB "src/tracking_ext/tracking/natnet/lib/x64/NatNetLib.dll")
      set(TRACKING_NATNET_IMPORT_LIB "src/tracking_ext/tracking/natnet/lib/x64/NatNetLib.lib")

      add_external_project(tracking_ext
          GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/mm-tracking
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

    endif()

  elseif(NAME STREQUAL "sim_sort")
    if(TARGET sim_sort)
      return()
    endif()
    
    add_external_project(sim_sort_ext
      GIT_REPOSITORY https://github.com/alexstraub1990/simultaneous-sort.git
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      TEST_COMMAND ""
    )
    add_external_library(sim_sort INTERFACE
      DEPENDS sim_sort_ext
      INCLUDE_DIR "src/sim_sort_ext/include"
    )
	
  else()
    message(FATAL_ERROR "Unknown external required \"${NAME}\"")
  endif()
endfunction(require_external)
