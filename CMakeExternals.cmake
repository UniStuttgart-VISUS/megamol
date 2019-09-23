#
# Centralized function to require externals to add them once by invoking
# require_external(<EXTERNAL_TARGET>).
#
# Think of this function as a big switch, testing for the name and presence 
# of the external target to guard against duplicated targets.
#
function(require_external NAME)
  set(FETCHCONTENT_QUIET ON CACHE BOOL "")
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "")

  # Header-only libraries #####################################################

  # Delaunator
  if(NAME STREQUAL "Delaunator")
    if(TARGET Delaunator)
      return()
    endif()

    add_external_headeronly_project(Delaunator
      GIT_REPOSITORY https://github.com/delfrrr/delaunator-cpp.git
      GIT_TAG "v0.4.0"
      INCLUDE_DIR "include")

  # Eigen
  elseif(NAME STREQUAL "Eigen")
    if(TARGET Eigen)
      return()
    endif()

    add_external_headeronly_project(Eigen
      GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
      GIT_TAG "3.3.4")

  # glm
  elseif(NAME STREQUAL "glm")
    if(TARGET glm)
      return()
    endif()

    add_external_headeronly_project(glm
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG "0.9.8")

  # glowl
  elseif(NAME STREQUAL "glowl")
    if(TARGET glowl)
      return()
    endif()

    add_external_headeronly_project(glowl
      GIT_REPOSITORY https://github.com/invor/glowl.git
      GIT_TAG "v0.3"
      INCLUDE_DIR "include")

  # json
  elseif(NAME STREQUAL "json")
    if(TARGET json)
      return()
    endif()

    add_external_headeronly_project(json
      GIT_REPOSITORY https://github.com/azadkuh/nlohmann_json_release.git
      GIT_TAG "v3.5.0")

  # libcxxopts
  elseif(NAME STREQUAL "libcxxopts")
    if(TARGET libcxxopts)
      return()
    endif()

    add_external_headeronly_project(libcxxopts
      DEPENDS libzmq
      GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
      GIT_TAG "v2.1.1"
      INCLUDE_DIR "include")

  # mmpld_io
  elseif(NAME STREQUAL "mmpld_io")
    if(TARGET mmpld_io)
      return()
    endif()

    add_external_headeronly_project(mmpld_io
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/mmpld_io.git
      INCLUDE_DIR "include")

  # nanoflann
  elseif(NAME STREQUAL "nanoflann")
    if(TARGET nanoflann)
      return()
    endif()

    add_external_headeronly_project(nanoflann
      GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
      GIT_TAG "v1.3.0"
      INCLUDE_DIR "include")

  # Built libraries #####################################################

  # adios2
  elseif(NAME STREQUAL "adios2")
    if(TARGET adios2)
      return()
    endif()

    if(WIN32)
      set(ADIOS2_IMPORT_LIB "lib/adios2.lib")
      set(ADIOS2_LIB "bin/adios2.dll")
    else()
      include(GNUInstallDirs)
      set(ADIOS2_IMPORT_LIB "${CMAKE_INSTALL_LIBDIR}/libadios2.so")
      set(ADIOS2_LIB "${CMAKE_INSTALL_LIBDIR}/libadios2.so")
    endif()

    add_external_project(adios2
      GIT_REPOSITORY https://github.com/ornladios/ADIOS2.git
      GIT_TAG "v2.3.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ADIOS2_IMPORT_LIB}"
      CMAKE_ARGS 
        -DBUILD_TESTING=OFF -DADIOS2_USE_BZip2=OFF 
        -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_HDF5=OFF 
        -DADIOS2_USE_Python=OFF -DADIOS2_USE_SST=OFF 
        -DADIOS2_USE_SZ=OFF -DADIOS2_USE_SysVShMem=OFF 
        -DADIOS2_USE_ZFP=OFF -DADIOS2_USE_ZeroMQ=OFF 
        -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

    add_external_library(adios2 SHARED
      IMPORT_LIBRARY ${ADIOS2_IMPORT_LIB}
      LIBRARY ${ADIOS2_LIB})

  # bhtsne
  elseif(NAME STREQUAL "bhtsne")
    if(TARGET bhtsne)
      return()
    endif()

    set(BHTSNE_LIB_DEBUG "lib/${CMAKE_STATIC_LIBRARY_PREFIX}bhtsned${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(BHTSNE_LIB_RELEASE "lib/${CMAKE_STATIC_LIBRARY_PREFIX}bhtsne${CMAKE_STATIC_LIBRARY_SUFFIX}")

    add_external_project(bhtsne
      GIT_REPOSITORY https://github.com/lvdmaaten/bhtsne.git
      GIT_TAG "36b169c88250d0afe51828448dfdeeaa508f13bc"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}bhtsne<SUFFIX>${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DEBUG_SUFFIX d
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/bhtsne/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    add_external_library(bhtsne STATIC
      LIBRARY_DEBUG ${BHTSNE_LIB_DEBUG}
      LIBRARY_RELEASE ${BHTSNE_LIB_RELEASE})

  # glfw3
  elseif(NAME STREQUAL "glfw3")
    if(TARGET glfw3)
      return()
    endif()

    if (MSVC)
      set(GLFW_PRODUCT "lib/glfw3dll.lib")
      set(GLFW_IMPORT_LIBRARY "lib/glfw3dll.lib")
      set(GLFW_LIBRARY "lib/glfw3.dll")
    else()
      set(GLFW_PRODUCT "lib/libglfw.so")
      set(GLFW_IMPORT_LIBRARY "")
      set(GLFW_LIBRARY "lib/libglfw.so")
    endif()

    add_external_project(glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG "3.2.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${GLFW_PRODUCT}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DGLFW_BUILD_EXAMPLES=OFF
        -DGLFW_BUILD_TESTS=OFF
        -DGLFW_BUILD_DOCS=OFF)

    add_external_library(glfw3 SHARED
      PROJECT glfw
      IMPORT_LIBRARY ${GLFW_IMPORT_LIBRARY}
      LIBRARY ${GLFW_LIBRARY})

  # IceT
  elseif(NAME STREQUAL "IceT")
    if(TARGET IceTCore)
      return()
    endif()

    if(WIN32)
      set(ICET_CORE_PRODUCT "lib/IceTCore.lib")
      set(ICET_CORE_IMPORT_LIB "lib/IceTCore.lib")
      set(ICET_CORE_LIB "bin/IceTCore.dll")
      set(ICET_GL_PRODUCT "lib/IceTGL.lib")
      set(ICET_GL_IMPORT_LIB "lib/IceTGL.lib")
      set(ICET_GL_LIB "bin/IceTGL.dll")
      set(ICET_MPI_PRODUCT "lib/IceTMPI.lib")
      set(ICET_MPI_IMPORT_LIB "lib/IceTMPI.lib")
      set(ICET_MPI_LIB "bin/IceTMPI.dll")
    else()
      include(GNUInstallDirs)
      set(ICET_CORE_PRODUCT "lib/libIceTCore.so")
      set(ICET_CORE_LIB "lib/libIceTCore.so")
      set(ICET_GL_PRODUCT "lib/libIceTGL.so")
      set(ICET_GL_LIB "lib/libIceTGL.so")
      set(ICET_MPI_PRODUCT "lib/libIceTMPI.so")
      set(ICET_MPI_LIB "lib/libIceTMPI.so")
    endif()
    
    add_external_project(IceT
      GIT_REPOSITORY https://gitlab.kitware.com/icet/icet.git
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ICET_CORE_PRODUCT}" "<INSTALL_DIR>/${ICET_GL_PRODUCT}" "<INSTALL_DIR>/${ICET_MPI_PRODUCT}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DICET_BUILD_TESTING=OFF
        -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

    add_external_library(IceTCore SHARED
      PROJECT IceT
      IMPORT_LIBRARY_DEBUG ${ICET_CORE_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${ICET_CORE_IMPORT_LIB}
      LIBRARY_DEBUG ${ICET_CORE_LIB}
      LIBRARY_RELEASE ${ICET_CORE_LIB})

    add_external_library(IceTGL SHARED
      PROJECT IceT
      IMPORT_LIBRARY_DEBUG ${ICET_GL_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${ICET_GL_IMPORT_LIB}
      LIBRARY_DEBUG ${ICET_GL_LIB}
      LIBRARY_RELEASE ${ICET_GL_LIB})

    add_external_library(IceTMPI SHARED
      PROJECT IceT
      IMPORT_LIBRARY_DEBUG ${ICET_MPI_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${ICET_MPI_IMPORT_LIB}
      LIBRARY_DEBUG ${ICET_MPI_LIB}
      LIBRARY_RELEASE ${ICET_MPI_LIB})

  # imgui
  elseif(NAME STREQUAL "imgui")
    if(TARGET imgui)
      return()
    endif()

    set(IMGUI_LIB "lib/${CMAKE_STATIC_LIBRARY_PREFIX}imgui${CMAKE_STATIC_LIBRARY_SUFFIX}")

    add_external_project(imgui
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG "v1.70"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${IMGUI_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/imgui/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    external_get_property(imgui SOURCE_DIR)

    add_external_library(imgui STATIC
      LIBRARY ${IMGUI_LIB})

    target_include_directories(imgui INTERFACE "${SOURCE_DIR}/examples" "${SOURCE_DIR}/misc/cpp")

    set(imgui_files
      "${SOURCE_DIR}/examples/imgui_impl_opengl3.cpp"
      "${SOURCE_DIR}/examples/imgui_impl_opengl3.h"
      "${SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp"
      "${SOURCE_DIR}/misc/cpp/imgui_stdlib.h"
      PARENT_SCOPE)

  # libpng
  elseif(NAME STREQUAL "libpng")
    if(TARGET libpng)
      return()
    endif()

    require_external(zlib)

    if(MSVC)
      set(LIBPNG_PRODUCT "lib/libpng16_static<SUFFIX>${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(LIBPNG_DEBUG "lib/libpng16_staticd${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(LIBPNG_RELEASE "lib/libpng16_static${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
      include(GNUInstallDirs)
      set(LIBPNG_PRODUCT "${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}png16<SUFFIX>${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(LIBPNG_DEBUG "${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}png16d${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(LIBPNG_RELEASE "${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}png16${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()

    if(MSVC)
      set(ZLIB_LIBRARY "lib/zlibstatic<SUFFIX>${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
      include(GNUInstallDirs)
      set(ZLIB_LIBRARY "lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()

    external_get_property(zlib INSTALL_DIR)

    add_external_project(libpng
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/libpng.git
      GIT_TAG "v1.6.34"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${LIBPNG_PRODUCT}"
      DEBUG_SUFFIX d
      DEPENDS zlib_ext
      CMAKE_ARGS
        -DPNG_BUILD_ZLIB=ON
        -DPNG_SHARED=OFF
        -DPNG_TESTS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DZLIB_LIBRARY:PATH=${INSTALL_DIR}/${ZLIB_LIBRARY}
        -DZLIB_INCLUDE_DIR:PATH=${INSTALL_DIR}/include
        -DZLIB_VERSION_STRING:STRING=${ZLIB_VERSION_STRING}
        -DZLIB_VERSION_MAJOR:STRING=${ZLIB_VERSION_MAJOR}
        -DZLIB_VERSION_MINOR:STRING=${ZLIB_VERSION_MINOR}
        -DZLIB_VERSION_PATCH:STRING=${ZLIB_VERSION_PATCH}
        -DZLIB_VERSION_TWEAK:STRING=${ZLIB_VERSION_TWEAK}
        -DZLIB_MAJOR_VERSION:STRING=${ZLIB_VERSION_MAJOR}
        -DZLIB_MINOR_VERSION:STRING=${ZLIB_VERSION_MINOR}
        -DZLIB_PATCH_VERSION:STRING=${ZLIB_VERSION_PATCH})

    add_external_library(libpng STATIC
      LIBRARY_DEBUG ${LIBPNG_DEBUG}
      LIBRARY_RELEASE ${LIBPNG_RELEASE}
      INTERFACE_LIBRARIES zlib)

  # libzmq / libcppzmq
  elseif(NAME STREQUAL "libzmq" OR NAME STREQUAL "libcppzmq")
    if(TARGET libzmq)
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
      set(ZMQ_PRODUCT "lib/libzmq${MSVC_TOOLSET}-mt<SUFFIX>-${ZMQ_VER}.lib")
      set(ZMQ_IMPORT_DEBUG "lib/libzmq${MSVC_TOOLSET}-mt-gd-${ZMQ_VER}.lib")
      set(ZMQ_IMPORT_RELEASE "lib/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.lib")
      set(ZMQ_DEBUG "bin/libzmq${MSVC_TOOLSET}-mt-gd-${ZMQ_VER}.dll")
      set(ZMQ_RELEASE "bin/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.dll")
    else()
      include(GNUInstallDirs)
      set(ZMQ_PRODUCT "${CMAKE_INSTALL_LIBDIR}/libzmq.so")
      set(ZMQ_IMPORT_DEBUG "")
      set(ZMQ_IMPORT_RELEASE "")
      set(ZMQ_DEBUG "${CMAKE_INSTALL_LIBDIR}/libzmq.so")
      set(ZMQ_RELEASE ${ZMQ_DEBUG})
    endif()

    add_external_project(libzmq
      GIT_REPOSITORY https://github.com/zeromq/libzmq.git
      GIT_TAG 56ace6d03f521b9abb5a50176ec7763c1b77afa9
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZMQ_PRODUCT}"
      DEBUG_SUFFIX -gd
      CMAKE_ARGS
        -DZMQ_BUILD_TESTS=OFF
        -DENABLE_PRECOMPILED=OFF)

    add_external_library(libzmq SHARED
      IMPORT_LIBRARY_DEBUG ${ZMQ_IMPORT_DEBUG}
      IMPORT_LIBRARY_RELEASE ${ZMQ_IMPORT_RELEASE}
      LIBRARY_DEBUG ${ZMQ_DEBUG}
      LIBRARY_RELEASE ${ZMQ_RELEASE})

    add_external_headeronly_project(libcppzmq
      DEPENDS libzmq
      GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
      GIT_TAG "v4.4.1")

  # quickhull
  elseif(NAME STREQUAL "quickhull")
    if(TARGET quickhull)
      return()
    endif()

    if(WIN32)
      set(QUICKHULL_IMPORT_LIB "lib/quickhull.lib")
      set(QUICKHULL_LIB "bin/quickhull.dll")
      set(QUICKHULL_CMAKE_ARGS "")
    else()
      set(QUICKHULL_IMPORT_LIB "lib/libquickhull.so")
      set(QUICKHULL_LIB "lib/libquickhull.so")
      set(QUICKHULL_CMAKE_ARGS -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC")
    endif()

    add_external_project(quickhull
      GIT_REPOSITORY https://github.com/akuukka/quickhull.git
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${QUICKHULL_IMPORT_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/quickhull/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt"
      CMAKE_ARGS
        ${QUICKHULL_CMAKE_ARGS})

    add_external_library(quickhull SHARED
      IMPORT_LIBRARY ${QUICKHULL_IMPORT_LIB}
      LIBRARY ${QUICKHULL_LIB})

  # snappy
  elseif(NAME STREQUAL "snappy")
    if(TARGET snappy)
      return()
    endif()

    if(WIN32)
      set(SNAPPY_PRODUCT "lib/snappy.lib")
      set(SNAPPY_IMPORT_LIB "lib/snappy.lib")
      set(SNAPPY_LIB "bin/snappy.dll")
    else()
      include(GNUInstallDirs)
      set(SNAPPY_PRODUCT "lib/libsnappy.so")
      set(SNAPPY_LIB "${CMAKE_INSTALL_LIBDIR}/libsnappy.so")
    endif()

    add_external_project(snappy
      GIT_REPOSITORY https://github.com/google/snappy.git
      GIT_TAG "1.1.7"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${SNAPPY_PRODUCT}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DSNAPPY_BUILD_TESTS=OFF
        -DCMAKE_BUILD_TYPE=Release)

    add_external_library(snappy SHARED
      IMPORT_LIBRARY_DEBUG ${SNAPPY_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${SNAPPY_IMPORT_LIB}
      LIBRARY_DEBUG ${SNAPPY_LIB}
      LIBRARY_RELEASE ${SNAPPY_LIB})

  # tinyobjloader
  elseif(NAME STREQUAL "tinyobjloader")
    if(TARGET tinyobjloader)
      return()
    endif()

    if(WIN32)
      set(TINYOBJLOADER_LIB "lib/tinyobjloader.lib")
    else()
      include(GNUInstallDirs)
      set(TINYOBJLOADER_LIB "${CMAKE_INSTALL_LIBDIR}/libtinyobjloader.a")
    endif()

    add_external_project(tinyobjloader
      GIT_REPOSITORY https://github.com/syoyo/tinyobjloader.git
      GIT_TAG "v2.0.0-rc1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TINYOBJLOADER_LIB}"
      CMAKE_ARGS
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC)

    add_external_library(tinyobjloader STATIC
      LIBRARY ${TINYOBJLOADER_LIB})

  # tinyply
  elseif(NAME STREQUAL "tinyply")
    if(TARGET tinyply)
      return()
    endif()

    if(WIN32)
      set(TNY_PRODUCT "lib/tinyply<SUFFIX>.lib")
      set(TNY_IMPORT_LIB "lib/tinyply.lib")
      set(TNY_IMPORT_LIB_DEBUG "lib/tinyplyd.lib")
      set(TNY_LIB "bin/tinyply.dll")
      set(TNY_LIB_DEBUG "bin/tinyplyd.dll")
    else()
      include(GNUInstallDirs)
      set(TNY_PRODUCT "lib/libtinyply<SUFFIX>.so")
      set(TNY_IMPORT_LIB_DEBUG "lib/libtinyplyd.so")
      set(TNY_IMPORT_LIB "lib/libtinyply.so")
      set(TNY_LIB_DEBUG "lib/libtinyplyd.so")
      set(TNY_LIB "lib/libtinyply.so")
    endif()

    add_external_project(tinyply
      GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
      GIT_TAG "2.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TNY_PRODUCT}"
      DEBUG_SUFFIX d
      CMAKE_ARGS
        -DSHARED_LIB=true)

    add_external_library(tinyply SHARED
      IMPORT_LIBRARY_DEBUG ${TNY_IMPORT_LIB_DEBUG}
      IMPORT_LIBRARY_RELEASE ${TNY_IMPORT_LIB}
      LIBRARY_DEBUG ${TNY_LIB_DEBUG}
      LIBRARY_RELEASE ${TNY_LIB})

  # tracking
  elseif(NAME STREQUAL "tracking")
    if(TARGET tracking)
      return()
    endif()

    set(TRACKING_LIB "bin/tracking.dll")
    set(TRACKING_IMPORT_LIB "lib/tracking.lib")
    set(TRACKING_NATNET_LIB "src/tracking_ext/tracking/natnet/lib/x64/NatNetLib.dll")
    set(TRACKING_NATNET_IMPORT_LIB "src/tracking_ext/tracking/natnet/lib/x64/NatNetLib.lib")

    add_external_project(tracking
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/mm-tracking
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TRACKING_IMPORT_LIB}" "<INSTALL_DIR>/${TRACKING_NATNET_IMPORT_LIB}"
      CMAKE_ARGS
        -DCREATE_TRACKING_TEST_PROGRAM=OFF)

    add_external_library(tracking SHARED 
      IMPORT_LIBRARY_DEBUG ${TRACKING_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${TRACKING_IMPORT_LIB}
      LIBRARY_DEBUG ${TRACKING_LIB}
      LIBRARY_RELEASE ${TRACKING_LIB})

    add_external_library(natnet SHARED 
      PROJECT tracking
      IMPORT_LIBRARY_DEBUG ${TRACKING_NATNET_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${TRACKING_NATNET_IMPORT_LIB}
      LIBRARY_DEBUG ${TRACKING_NATNET_LIB}
      LIBRARY_RELEASE ${TRACKING_NATNET_LIB})

    add_external_library(tracking_int INTERFACE
      PROJECT tracking
      INCLUDE_DIR "src/tracking_ext/tracking/include")

  # zfp
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

    add_external_project(zfp
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

    add_external_library(zfp STATIC
      LIBRARY ${ZFP_LIB})

  # zlib
  elseif(NAME STREQUAL "zlib")
    if(TARGET zlib)
      return()
    endif()

    if(MSVC)
      set(ZLIB_PRODUCT "lib/zlibstatic<SUFFIX>${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(ZLIB_DEBUG "lib/zlibstaticd${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(ZLIB_RELEASE "lib/zlibstatic${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
      include(GNUInstallDirs)
      set(ZLIB_PRODUCT "lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(ZLIB_DEBUG "lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(ZLIB_RELEASE "lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()

    add_external_project(zlib
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG "v1.2.11"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZLIB_PRODUCT}"
      DEBUG_SUFFIX d
      CMAKE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON)

    add_external_library(zlib STATIC
      LIBRARY_DEBUG ${ZLIB_DEBUG}
      LIBRARY_RELEASE ${ZLIB_RELEASE})

    set(ZLIB_VERSION_STRING "1.2.11" CACHE STRING "" FORCE)
    set(ZLIB_VERSION_MAJOR 1 CACHE STRING "" FORCE)
    set(ZLIB_VERSION_MINOR 2 CACHE STRING "" FORCE)
    set(ZLIB_VERSION_PATCH 11 CACHE STRING "" FORCE)
    set(ZLIB_VERSION_TWEAK "" CACHE STRING "" FORCE)

    mark_as_advanced(FORCE ZLIB_VERSION_STRING)
    mark_as_advanced(FORCE ZLIB_VERSION_MAJOR)
    mark_as_advanced(FORCE ZLIB_VERSION_MINOR)
    mark_as_advanced(FORCE ZLIB_VERSION_PATCH)
    mark_as_advanced(FORCE ZLIB_VERSION_TWEAK)

  else()
    message(FATAL_ERROR "Unknown external required \"${NAME}\"")
  endif()

  mark_as_advanced(FORCE FETCHCONTENT_BASE_DIR)
  mark_as_advanced(FORCE FETCHCONTENT_FULLY_DISCONNECTED)
  mark_as_advanced(FORCE FETCHCONTENT_QUIET)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED)
endfunction(require_external)
