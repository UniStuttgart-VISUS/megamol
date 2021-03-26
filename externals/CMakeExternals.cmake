# Clone external script
if(NOT EXISTS "${CMAKE_BINARY_DIR}/script-externals")
  message(STATUS "Downloading external scripts")
  execute_process(COMMAND
    ${GIT_EXECUTABLE} clone -b v2.3 https://github.com/UniStuttgart-VISUS/megamol-cmake-externals.git script-externals --depth 1
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Include external script
include("${CMAKE_BINARY_DIR}/script-externals/cmake/External.cmake")

#
# Centralized function to require externals to add them once by invoking
# require_external(<EXTERNAL_TARGET>).
#
# Think of this function as a big switch, testing for the name and presence 
# of the external target to guard against duplicated targets.
#
function(require_external NAME)
  set(FETCHCONTENT_QUIET ON CACHE BOOL "")

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
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG "3.3.7")

  # glm
  elseif(NAME STREQUAL "glm")
    if(TARGET glm)
      return()
    endif()

    add_external_headeronly_project(glm
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG "0.9.9.8")

  # glowl
  elseif(NAME STREQUAL "glowl")
    if(TARGET glowl)
      return()
    endif()

    add_external_headeronly_project(glowl
      GIT_REPOSITORY https://github.com/invor/glowl.git
      GIT_TAG "v0.4e"
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
      # we are waiting for v3 which brings allowing unrecognized options
      #GIT_TAG "v2.1.1"
      GIT_TAG "dd45a0801c99d62109aaa29f8c410ba8def2fbf2"
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

  # tinygltf
  elseif(NAME STREQUAL "tinygltf")
    if(TARGET tinygltf)
      return()
    endif()

    add_external_headeronly_project(tinygltf
      GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
      GIT_TAG "v2.2.0")

  elseif(NAME STREQUAL "sim_sort")
    if(TARGET sim_sort)
      return()
    endif()

    add_external_headeronly_project(sim_sort
      GIT_REPOSITORY https://github.com/alexstraub1990/simultaneous-sort.git
      INCLUDE_DIR "include")

  # Built libraries #####################################################

  # adios2
  elseif(NAME STREQUAL "adios2")
    if(TARGET adios2)
      return()
    endif()

    if(WIN32)
      set(ADIOS2_LIB "lib/adios2.lib")
    else()
      include(GNUInstallDirs)
      set(ADIOS2_LIB "${CMAKE_INSTALL_LIBDIR}/libadios2.a")
    endif()

    add_external_project(adios2 STATIC
      GIT_REPOSITORY https://github.com/ornladios/ADIOS2.git
      GIT_TAG "v2.4.0"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ADIOS2_LIB}"
      CMAKE_ARGS 
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_TESTING=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DADIOS2_USE_BZip2=OFF
        -DADIOS2_USE_Fortran=OFF
        -DADIOS2_USE_HDF5=OFF
        -DADIOS2_USE_PNG=OFF
        -DADIOS2_USE_Python=OFF
        -DADIOS2_USE_SST=OFF
        -DADIOS2_USE_SZ=OFF
        -DADIOS2_USE_SysVShMem=OFF
        -DADIOS2_USE_ZFP=OFF
        -DADIOS2_USE_ZeroMQ=OFF
        -DADIOS2_USE_Profiling=OFF
        -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

    add_external_library(adios2
      LIBRARY ${ADIOS2_LIB})

  # libigl
  elseif(NAME STREQUAL "libigl")
    if(TARGET libigl)
      return()
    endif()

    if(WIN32)
      set(LIBIGL_LIB "")
    else()
      include(GNUInstallDirs)
      set(LIBIGL_LIB "")
    endif()

    add_external_headeronly_project(libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG "v2.1.0"
        INCLUDE_DIR "include")


  # bhtsne
  elseif(NAME STREQUAL "bhtsne")
    if(TARGET bhtsne)
      return()
    endif()

    if(WIN32)
      set(BHTSNE_LIB "lib/bhtsne.lib")
    else()
      set(BHTSNE_LIB "lib/libbhtsne.a")
    endif()

    add_external_project(bhtsne STATIC
      GIT_REPOSITORY https://github.com/lvdmaaten/bhtsne.git
      GIT_TAG "36b169c88250d0afe51828448dfdeeaa508f13bc"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${BHTSNE_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/bhtsne/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    add_external_library(bhtsne
      LIBRARY ${BHTSNE_LIB})

  # fmt
  elseif(NAME STREQUAL "fmt")
    if(TARGET fmt)
      return()
    endif()

    if(WIN32)
      set(FMT_LIB "lib/fmt<SUFFIX>.lib")
    else()
      include(GNUInstallDirs)
      set(FMT_LIB "${CMAKE_INSTALL_LIBDIR}/libfmt<SUFFIX>.a")
    endif()

    add_external_project(fmt STATIC
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG "6.2.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${FMT_LIB}"
      DEBUG_SUFFIX "d"
      CMAKE_ARGS
        -DFMT_DOC=OFF
        -DFMT_TEST=OFF
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC)

    add_external_library(fmt
      LIBRARY ${FMT_LIB}
      DEBUG_SUFFIX "d")

  #glad 
  elseif(NAME STREQUAL "glad")
    if(TARGET glad)
      return()
    endif()

    include(GNUInstallDirs)

    if(WIN32)
      set(GLAD_LIB "lib/glad.lib")
    else()
      set(GLAD_LIB "lib/libglad.a")
    endif()

    add_external_project(glad STATIC
      SOURCE_DIR glad
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${GLAD_LIB}")

    add_external_library(glad
      PROJECT glad
      LIBRARY ${GLAD_LIB})

  # glfw3
  elseif(NAME STREQUAL "glfw3")
    if(TARGET glfw3)
      return()
    endif()

    include(GNUInstallDirs)

    if (WIN32)
      set(GLFW_LIB "${CMAKE_INSTALL_LIBDIR}/glfw3.lib")
    else ()
      set(GLFW_LIB "${CMAKE_INSTALL_LIBDIR}/libglfw3.a")
    endif ()

    add_external_project(glfw STATIC
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG "3.3.2"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${GLFW_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DGLFW_BUILD_DOCS=OFF
        -DGLFW_BUILD_EXAMPLES=OFF
        -DGLFW_BUILD_TESTS=OFF)

    add_external_library(glfw3
      PROJECT glfw
      LIBRARY ${GLFW_LIB})

  # IceT
  elseif(NAME STREQUAL "IceT")
    if(TARGET IceTCore)
      return()
    endif()

    if(WIN32)
      set(ICET_CORE_LIB "lib/IceTCore.lib")
      set(ICET_GL_LIB "lib/IceTGL.lib")
      set(ICET_MPI_LIB "lib/IceTMPI.lib")
    else()
      include(GNUInstallDirs)
      set(ICET_CORE_LIB "lib/libIceTCore.a")
      set(ICET_GL_LIB "lib/libIceTGL.a")
      set(ICET_MPI_LIB "lib/libIceTMPI.a")
    endif()
    
    add_external_project(IceT STATIC
      GIT_REPOSITORY https://gitlab.kitware.com/icet/icet.git
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

  # imgui
  elseif(NAME STREQUAL "imgui")
    if(NOT TARGET imgui)
      if(WIN32)
        set(IMGUI_LIB "lib/imgui.lib")
      else()
        set(IMGUI_LIB "lib/libimgui.a")
      endif()

      add_external_project(imgui STATIC
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG 455c21df7100a4727dd6e4c8e69249b7de21d24c # docking branch > version "1.79"
        BUILD_BYPRODUCTS "<INSTALL_DIR>/${IMGUI_LIB}"
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_SOURCE_DIR}/externals/imgui/CMakeLists.txt"
          "<SOURCE_DIR>/CMakeLists.txt")

      add_external_library(imgui
        LIBRARY ${IMGUI_LIB})
    endif()

    external_get_property(imgui SOURCE_DIR)

    target_include_directories(imgui INTERFACE "${SOURCE_DIR}/backends" "${SOURCE_DIR}/misc/cpp")

    set(imgui_files
      "${SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
      "${SOURCE_DIR}/backends/imgui_impl_opengl3.h"
      "${SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp"
      "${SOURCE_DIR}/misc/cpp/imgui_stdlib.h"
      PARENT_SCOPE)

  # imguizmoquat
  elseif(NAME STREQUAL "imguizmoquat")
    if(TARGET imguizmoquat)
      return()
    endif()

    require_external(imgui)

    if(WIN32)
      set(IMGUIZMOQUAT_LIB "lib/imguizmoquat.lib")
    else()
      set(IMGUIZMOQUAT_LIB "lib/libimguizmoquat.a")
    endif()

    if(WIN32)
      set(IMGUI_LIB "lib/imgui.lib")
    else()
      set(IMGUI_LIB "lib/libimgui.a")
    endif()

    external_get_property(imgui INSTALL_DIR)

    add_external_project(imguizmoquat STATIC
      GIT_REPOSITORY https://github.com/braunms/imGuIZMO.quat.git
      GIT_TAG "v3.0a"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${IMGUIZMOQUAT_LIB}"
      DEPENDS imgui
      CMAKE_ARGS
        -DIMGUI_LIBRARY:PATH=${INSTALL_DIR}/${IMGUI_LIB}
        -DIMGUI_INCLUDE_DIR:PATH=${INSTALL_DIR}/include
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_SOURCE_DIR}/externals/imguizmoquat/CMakeLists.txt"
          "<SOURCE_DIR>/CMakeLists.txt")

    add_external_library(imguizmoquat
        LIBRARY ${IMGUIZMOQUAT_LIB})

    external_get_property(imguizmoquat SOURCE_DIR)
    target_include_directories(imguizmoquat INTERFACE "${SOURCE_DIR}/imGuIZMO.quat")

  # libpng
  elseif(NAME STREQUAL "libpng")
    if(TARGET libpng)
      return()
    endif()

    require_external(zlib)

    if(MSVC)
      set(LIBPNG_LIB "lib/libpng16_static<SUFFIX>.lib")
    else()
      include(GNUInstallDirs)
      set(LIBPNG_LIB "${CMAKE_INSTALL_LIBDIR}/libpng16<SUFFIX>.a")
    endif()

    if(MSVC)
      set(ZLIB_LIB "lib/zlibstatic$<$<CONFIG:Debug>:d>.lib")
    else()
      include(GNUInstallDirs)
      set(ZLIB_LIB "lib/libz.a")
    endif()

    external_get_property(zlib INSTALL_DIR)

    add_external_project(libpng STATIC
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/libpng.git
      GIT_TAG "v1.6.34"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${LIBPNG_LIB}"
      DEBUG_SUFFIX d
      DEPENDS zlib
      CMAKE_ARGS
        -DPNG_BUILD_ZLIB=ON
        -DPNG_SHARED=OFF
        -DPNG_TESTS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DZLIB_LIBRARY:PATH=${INSTALL_DIR}/${ZLIB_LIB}
        -DZLIB_INCLUDE_DIR:PATH=${INSTALL_DIR}/include
        -DZLIB_VERSION_STRING:STRING=${ZLIB_VERSION_STRING}
        -DZLIB_VERSION_MAJOR:STRING=${ZLIB_VERSION_MAJOR}
        -DZLIB_VERSION_MINOR:STRING=${ZLIB_VERSION_MINOR}
        -DZLIB_VERSION_PATCH:STRING=${ZLIB_VERSION_PATCH}
        -DZLIB_VERSION_TWEAK:STRING=${ZLIB_VERSION_TWEAK}
        -DZLIB_MAJOR_VERSION:STRING=${ZLIB_VERSION_MAJOR}
        -DZLIB_MINOR_VERSION:STRING=${ZLIB_VERSION_MINOR}
        -DZLIB_PATCH_VERSION:STRING=${ZLIB_VERSION_PATCH})

    add_external_library(libpng
      LIBRARY ${LIBPNG_LIB}
      INTERFACE_LIBRARIES zlib
      DEBUG_SUFFIX d)

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

    include(GNUInstallDirs)

    if(WIN32)
      set(ZMQ_LIB "${CMAKE_INSTALL_LIBDIR}/libzmq${MSVC_TOOLSET}-mt-s<SUFFIX>-${ZMQ_VER}.lib")
    else()
      set(ZMQ_LIB "${CMAKE_INSTALL_LIBDIR}/libzmq.a")
    endif()

    add_external_project(libzmq STATIC
      GIT_REPOSITORY https://github.com/zeromq/libzmq.git
      GIT_TAG 56ace6d03f521b9abb5a50176ec7763c1b77afa9
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZMQ_LIB}"
      DEBUG_SUFFIX gd
      CMAKE_ARGS
        -DBUILD_SHARED=OFF
        -DBUILD_TESTS=OFF
        -DZMQ_BUILD_TESTS=OFF
        -DENABLE_PRECOMPILED=OFF
        -DWITH_DOCS=OFF)

    add_external_library(libzmq
      LIBRARY ${ZMQ_LIB}
      DEBUG_SUFFIX gd)

    set_target_properties(libzmq PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "ZMQ_STATIC")

    # TODO libzmq cmake does a lot more checks and options. This will probably work only in some configurations.
    if(WIN32)
      target_link_libraries(libzmq INTERFACE ws2_32 iphlpapi)
    endif()

    add_external_headeronly_project(libcppzmq
      DEPENDS libzmq
      GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
      GIT_TAG "v4.6.0")

  # lua
  elseif(NAME STREQUAL "lua")
    if(TARGET lua)
      return()
    endif()

    if(WIN32)
      set(LUA_LIB "lib/lua.lib")
    else()
      set(LUA_LIB "lib/liblua.a")
    endif()

    add_external_project(lua STATIC
      GIT_REPOSITORY https://github.com/lua/lua.git
      GIT_TAG v5.3.5
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${LUA_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/lua/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt"
        COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/lua/lua.hpp"
        "<SOURCE_DIR>/lua.hpp")

    add_external_library(lua
      LIBRARY ${LUA_LIB})

  # quickhull
  elseif(NAME STREQUAL "quickhull")
    if(TARGET quickhull)
      return()
    endif()

    if(WIN32)
      set(QUICKHULL_LIB "lib/quickhull.lib")
    else()
      set(QUICKHULL_LIB "lib/libquickhull.a")
    endif()

    add_external_project(quickhull STATIC
      GIT_REPOSITORY https://github.com/akuukka/quickhull.git
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${QUICKHULL_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/quickhull/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt"
      CMAKE_ARGS
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC)

    add_external_library(quickhull
      LIBRARY ${QUICKHULL_LIB})

  # snappy
  elseif(NAME STREQUAL "snappy")
    if(TARGET snappy)
      return()
    endif()

    if(WIN32)
      set(SNAPPY_LIB "lib/snappy.lib")
    else()
      include(GNUInstallDirs)
      set(SNAPPY_LIB "${CMAKE_INSTALL_LIBDIR}/libsnappy.a")
    endif()

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

  # spdlog
  elseif(NAME STREQUAL "spdlog")
    if(TARGET spdlog)
      return()
    endif()

    require_external(fmt)

    if(WIN32)
      set(SPDLOG_LIB "lib/spdlog<SUFFIX>.lib")
    else()
      include(GNUInstallDirs)
      set(SPDLOG_LIB "${CMAKE_INSTALL_LIBDIR}/libspdlog<SUFFIX>.a")
    endif()

    external_get_property(fmt BINARY_DIR)

    add_external_project(spdlog STATIC
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG "v1.7.0"
      DEPENDS fmt
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${SPDLOG_LIB}"
      DEBUG_SUFFIX "d"
      CMAKE_ARGS
        -DSPDLOG_BUILD_EXAMPLE=OFF
        -DSPDLOG_BUILD_TESTS=OFF
        -DSPDLOG_FMT_EXTERNAL=ON
        -Dfmt_DIR=${BINARY_DIR}
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC)

    add_external_library(spdlog
      LIBRARY ${SPDLOG_LIB}
      DEBUG_SUFFIX "d"
      DEPENDS fmt)

    target_compile_definitions(spdlog INTERFACE SPDLOG_FMT_EXTERNAL;SPDLOG_COMPILED_LIB)

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

    add_external_project(tinyobjloader STATIC
      GIT_REPOSITORY https://github.com/syoyo/tinyobjloader.git
      GIT_TAG "v2.0.0-rc1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TINYOBJLOADER_LIB}"
      CMAKE_ARGS
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC)

    add_external_library(tinyobjloader
      LIBRARY ${TINYOBJLOADER_LIB})

  # tinyply
  elseif(NAME STREQUAL "tinyply")
    if(TARGET tinyply)
      return()
    endif()

    include(GNUInstallDirs)

    if(WIN32)
      set(TNY_LIB "${CMAKE_INSTALL_LIBDIR}/tinyply<SUFFIX>.lib")
    else()
      set(TNY_LIB "${CMAKE_INSTALL_LIBDIR}/libtinyply<SUFFIX>.a")
    endif()

    add_external_project(tinyply STATIC
      GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
      GIT_TAG "2.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TNY_LIB}"
      DEBUG_SUFFIX d
      CMAKE_ARGS
        -DSHARED_LIB=OFF
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC)

    add_external_library(tinyply
      LIBRARY ${TNY_LIB}
      DEBUG_SUFFIX d)

  # tracking
  elseif(NAME STREQUAL "tracking")
    if(TARGET tracking)
      return()
    endif()

    if(NOT WIN32)
      message(WARNING "External 'tracking' requested, but not available on non-Windows systems")
    endif()

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

  # zlib
  elseif(NAME STREQUAL "zlib")
    if(TARGET zlib)
      return()
    endif()

    if(MSVC)
      set(ZLIB_LIB "lib/zlibstatic<SUFFIX>.lib")
    else()
      include(GNUInstallDirs)
      set(ZLIB_LIB "lib/libz.a")
    endif()

    add_external_project(zlib STATIC
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG "v1.2.11"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZLIB_LIB}"
      DEBUG_SUFFIX d
      CMAKE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON)

    add_external_library(zlib
      LIBRARY ${ZLIB_LIB}
      DEBUG_SUFFIX d)

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

  # vtkm
  elseif(NAME STREQUAL "vtkm")
    if(TARGET vtkm)
      return()
    endif()

    set(VTKM_VER 1.4)

    if(WIN32)
      set(VTKM_LIB_CONT "lib/vtkm_cont-${VTKM_VER}.lib")
      set(VTKM_LIB_DEBUG_CONT "lib/vtkm_cont-${VTKM_VER}.lib")
      set(VTKM_LIB_RENDERER "lib/vtkm_rendering-${VTKM_VER}.lib")
      set(VTKM_LIB_DEBUG_RENDERER "lib/vtkm_rendering-${VTKM_VER}.lib")
      set(VTKM_LIB_WORKLET "lib/vtkm_worklet-${VTKM_VER}.lib")
      set(VTKM_LIB_DEBUG_WORKLET "lib/vtkm_worklet-${VTKM_VER}.lib")
    else()
      include(GNUInstallDirs)
      set(VTKM_LIB_CONT "lib/vtkm_cont-${VTKM_VER}.a")
      set(VTKM_LIB_DEBUG_CONT "lib/vtkm_cont-${VTKM_VER}.a")
      set(VTKM_LIB_RENDERER "lib/vtkm_rendering-${VTKM_VER}.a")
      set(VTKM_LIB_DEBUG_RENDERER "lib/vtkm_rendering-${VTKM_VER}.a")
      set(VTKM_LIB_WORKLET "lib/vtkm_worklet-${VTKM_VER}.a")
      set(VTKM_LIB_DEBUG_WORKLET "lib/vtkm_worklet-${VTKM_VER}.a")
    endif()

    option(vtkm_ENABLE_CUDA "Option to build vtkm with cuda enabled" OFF)
    
    add_external_project(vtkm
      GIT_REPOSITORY https://gitlab.kitware.com/vtk/vtk-m.git
      GIT_TAG "v1.4.0"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DVTKm_ENABLE_TESTING:BOOL=OFF
        -DVTKm_ENABLE_CUDA:BOOL=${vtkm_ENABLE_CUDA}
        -DBUILD_TESTING:BOOL=OFF
        -VTKm_ENABLE_DEVELOPER_FLAGS:BOOL=OFF
        -DCMAKE_BUILD_TYPE=Release
        )

    add_external_library(vtkm_cont STATIC
      PROJECT vtkm
      LIBRARY_RELEASE "${VTKM_LIB_CONT}"
      LIBRARY_DEBUG "${VTKM_LIB_DEBUG_CONT}")

    add_external_library(vtkm_renderer STATIC
      PROJECT vtkm
      LIBRARY_RELEASE "${VTKM_LIB_RENDERER}"
      LIBRARY_DEBUG "${VTKM_LIB_DEBUG_RENDERER}")

    add_external_library(vtkm_worklet STATIC
      PROJECT vtkm
      LIBRARY_RELEASE "${VTKM_LIB_WORKLET}"
      LIBRARY_DEBUG "${VTKM_LIB_DEBUG_WORKLET}")
  else()
    message(FATAL_ERROR "Unknown external required \"${NAME}\"")
  endif()

  mark_as_advanced(FORCE FETCHCONTENT_BASE_DIR)
  mark_as_advanced(FORCE FETCHCONTENT_FULLY_DISCONNECTED)
  mark_as_advanced(FORCE FETCHCONTENT_QUIET)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED)
endfunction(require_external)
