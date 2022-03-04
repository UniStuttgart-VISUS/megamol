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
    ${GIT_EXECUTABLE} clone -b v2.3 https://github.com/UniStuttgart-VISUS/megamol-cmake-externals.git script-externals --depth 1
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

  # asmjit
  if (NAME STREQUAL "asmjit")
    if (TARGET asmjit)
      return()
    endif ()

    add_external_headeronly_project(asmjit INTERFACE
      GIT_REPOSITORY https://github.com/asmjit/asmjit.git
      GIT_TAG "8474400e82c3ea65bd828761539e5d9b25f6bd83")

  # Delaunator
  elseif (NAME STREQUAL "Delaunator")
    if (TARGET Delaunator)
      return()
    endif ()

    add_external_headeronly_project(Delaunator
      GIT_REPOSITORY https://github.com/delfrrr/delaunator-cpp.git
      GIT_TAG "v0.4.0"
      INCLUDE_DIR "include")

  # Eigen
  elseif (NAME STREQUAL "Eigen")
    if (TARGET Eigen)
      return()
    endif ()

    add_external_headeronly_project(Eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG "3.3.7")

  # glm
  elseif (NAME STREQUAL "glm")
    if (TARGET glm)
      return()
    endif ()

    add_external_headeronly_project(glm
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG "0.9.9.8")

  # glowl
  elseif (NAME STREQUAL "glowl")
    if (TARGET glowl)
      return()
    endif ()

    add_external_headeronly_project(glowl
      GIT_REPOSITORY https://github.com/invor/glowl.git
      GIT_TAG "dafee75f11c5d759df30ff651d6763e4e674dd0e"
      INCLUDE_DIR "include")
    target_compile_definitions(glowl INTERFACE GLOWL_OPENGL_INCLUDE_GLAD2)

  # json
  elseif (NAME STREQUAL "json")
    if (TARGET json)
      return()
    endif ()

    # The repo at https://github.com/nlohmann/json is too big, add local copy to avoid very slow download!
    add_external_headeronly_project(json
      SOURCE_DIR json)
    if (MSVC)
      target_sources(json INTERFACE "${CMAKE_SOURCE_DIR}/externals/json/nlohmann_json.natvis")
    endif ()

  # libcxxopts
  elseif (NAME STREQUAL "libcxxopts")
    if (TARGET libcxxopts)
      return()
    endif ()

    add_external_headeronly_project(libcxxopts
      GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
      # we are waiting for v3 which brings allowing unrecognized options
      #GIT_TAG "v2.1.1"
      GIT_TAG "dd45a0801c99d62109aaa29f8c410ba8def2fbf2"
      INCLUDE_DIR "include")

  # mmpld_io
  elseif (NAME STREQUAL "mmpld_io")
    if (TARGET mmpld_io)
      return()
    endif ()

    add_external_headeronly_project(mmpld_io
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/mmpld_io.git
      GIT_TAG 0002c64e0be4dddc137e4fe37db4b96361bc79bd
      INCLUDE_DIR "include")

  # nanoflann
  elseif (NAME STREQUAL "nanoflann")
    if (TARGET nanoflann)
      return()
    endif ()

    add_external_headeronly_project(nanoflann
      GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
      GIT_TAG "v1.3.0"
      INCLUDE_DIR "include")

  # tinygltf
  elseif (NAME STREQUAL "tinygltf")
    if (TARGET tinygltf)
      return()
    endif ()

    add_external_headeronly_project(tinygltf
      GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
      GIT_TAG "v2.5.0")
    target_compile_definitions(tinygltf INTERFACE TINYGLTF_NO_INCLUDE_JSON)

  # sim_sort
  elseif (NAME STREQUAL "sim_sort")
    if (TARGET sim_sort)
      return()
    endif ()

    add_external_headeronly_project(sim_sort
      GIT_REPOSITORY https://github.com/alexstraub1990/simultaneous-sort.git
      GIT_TAG 220fdf37fec2d9d3e3f7674194544ee70eb93ee7 # master on 2021-07-26, because nothing was specified here.
      INCLUDE_DIR "include")

  # ###########################################################################
  # ### Built libraries #######################################################
  # ###########################################################################

  # adios2
  elseif (NAME STREQUAL "adios2")
    if (TARGET adios2)
      return()
    endif ()

    if (WIN32)
      set(ADIOS2_LIB "lib/adios2.lib")
    else ()
      set(ADIOS2_LIB "${CMAKE_INSTALL_LIBDIR}/libadios2.a")
    endif ()

    add_external_project(adios2 STATIC
      GIT_REPOSITORY https://github.com/ornladios/ADIOS2.git
      GIT_TAG "v2.5.0"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ADIOS2_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DADIOS2_BUILD_EXAMPLES=OFF
        -DADIOS2_BUILD_TESTING=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DADIOS2_USE_BZip2=OFF
        -DADIOS2_USE_Fortran=OFF
        -DADIOS2_USE_HDF5=OFF
        -DADIOS2_USE_PNG=OFF
        -DADIOS2_USE_Profiling=OFF
        -DADIOS2_USE_Python=OFF
        -DADIOS2_USE_SST=OFF
        -DADIOS2_USE_SZ=OFF
        -DADIOS2_USE_SysVShMem=OFF
        -DADIOS2_USE_ZFP=OFF
        -DADIOS2_USE_ZeroMQ=OFF
        -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

    add_external_library(adios2
      LIBRARY ${ADIOS2_LIB})

  # bhtsne
  elseif (NAME STREQUAL "bhtsne")
    if (TARGET bhtsne)
      return()
    endif ()

    if (WIN32)
      set(BHTSNE_LIB "lib/bhtsne.lib")
    else ()
      set(BHTSNE_LIB "lib/libbhtsne.a")
    endif ()

    add_external_project(bhtsne STATIC
      GIT_REPOSITORY https://github.com/lvdmaaten/bhtsne.git
      GIT_TAG "36b169c88250d0afe51828448dfdeeaa508f13bc"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${BHTSNE_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/bhtsne/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    add_external_library(bhtsne
      LIBRARY ${BHTSNE_LIB})

  # blend2d
  elseif (NAME STREQUAL "blend2d")
    if (TARGET blend2d)
      return()
    endif ()

    if (WIN32)
      set(BLEND2D_LIB "lib/blend2d.lib")
    else ()
      set(BLEND2D_LIB "lib/libblend2d.a")
    endif ()

    require_external(asmjit)
    external_get_property(asmjit SOURCE_DIR)

    add_external_project(blend2d STATIC
      GIT_REPOSITORY https://github.com/blend2d/blend2d.git
      GIT_TAG "8aeac6cb34b00898ae725bd76eb3bb2c7cffcf86"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${BLEND2D_IMPORT_LIB}" "<INSTALL_DIR>/${BLEND2D_LIB}"
      CMAKE_ARGS
        -DASMJIT_DIR=${SOURCE_DIR})

    add_external_library(blend2d
      DEPENDS asmjit
      INCLUDE_DIR "include"
      LIBRARY ${BLEND2D_LIB})

  # chemfiles
  elseif(NAME STREQUAL "chemfiles")
    if (TARGET chemfiles)
      return()
    endif()

    if (WIN32)
      set(CHEMFILES_LIB "lib/chemfiles.lib")
    else ()
      set(CHEMFILES_LIB "lib/chemfiles.a")
    endif ()

    add_external_project(chemfiles STATIC
      GIT_REPOSITORY https://github.com/chemfiles/chemfiles.git
      GIT_TAG "0.10.2"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${CHEMFILES_LIB}"
    )

    add_external_library(chemfiles
      INCLUDE_DIR "include"
      LIBRARY ${CHEMFILES_LIB})

  # Corsair CUE SDK
  elseif (NAME STREQUAL "CUESDK")
    if (TARGET CUESDK)
      return()
    endif ()

    FetchContent_Declare(
      cuesdk_archive
      URL https://github.com/CorsairOfficial/cue-sdk/releases/download/v3.0.378/CUESDK_3.0.378.zip)
    FetchContent_GetProperties(cuesdk_archive)
    if (NOT cuesdk_archive_POPULATED)
      FetchContent_Populate(cuesdk_archive)
      add_library(CUESDK SHARED IMPORTED GLOBAL)
      set_target_properties(CUESDK PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${cuesdk_archive_SOURCE_DIR}/include"
        IMPORTED_CONFIGURATIONS "Release"
        IMPORTED_LOCATION "${cuesdk_archive_SOURCE_DIR}/redist/x64/CUESDK.x64_2017.dll"
        IMPORTED_IMPLIB "${cuesdk_archive_SOURCE_DIR}/lib/x64/CUESDK.x64_2017.lib")
      install(DIRECTORY "${cuesdk_archive_SOURCE_DIR}/redist/x64/" DESTINATION "bin" FILES_MATCHING PATTERN "*2017.dll")
    endif ()

  # expat
  elseif (NAME STREQUAL "expat")
    if (TARGET expat)
      return()
    endif ()

    if (WIN32)
      set(EXPAT_LIB "lib/expat<SUFFIX>.lib")
    else ()
      set(EXPAT_LIB "lib/libexpat.a")
    endif ()

    # Files in core were originally at 64f3cf982a156a62c1fdb44d864144ee5871159e
    # This seems to be master at 07.06.2017, somewhere between 2.2.0 and 2.2.1
    add_external_project(expat STATIC
      GIT_REPOSITORY https://github.com/libexpat/libexpat
      GIT_TAG "R_2_2_1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${EXPAT_LIB}"
      SOURCE_SUBDIR "expat"
      DEBUG_SUFFIX "d"
      CMAKE_ARGS
        -DBUILD_doc=OFF
        -DBUILD_examples=OFF
        -DBUILD_shared=OFF
        -DBUILD_tests=OFF
        -DBUILD_tools=OFF)

    add_external_library(expat
      LIBRARY ${EXPAT_LIB}
      DEBUG_SUFFIX "d")

  # fmt
  elseif (NAME STREQUAL "fmt")
    if (TARGET fmt)
      return()
    endif ()

    if (WIN32)
      set(FMT_LIB "lib/fmt<SUFFIX>.lib")
    else ()
      set(FMT_LIB "${CMAKE_INSTALL_LIBDIR}/libfmt<SUFFIX>.a")
    endif ()

    add_external_project(fmt STATIC
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG "6.2.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${FMT_LIB}"
      DEBUG_SUFFIX "d"
      CMAKE_ARGS
        -DFMT_DOC=OFF
        -DFMT_TEST=OFF)

    add_external_library(fmt
      LIBRARY ${FMT_LIB}
      DEBUG_SUFFIX "d")

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

  # glfw
  elseif (NAME STREQUAL "glfw")
    if (TARGET glfw)
      return()
    endif ()

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

    add_external_library(glfw
      PROJECT glfw
      LIBRARY ${GLFW_LIB})

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

  # imgui
  elseif (NAME STREQUAL "imgui")
    if (TARGET imgui)
      return()
    endif ()

    if (ENABLE_GL)
      require_external(glfw)
      external_get_property(glfw INSTALL_DIR)
      set(glfw_include_dir "${INSTALL_DIR}/include")
      set(glad_include_dir "${CMAKE_SOURCE_DIR}/externals/glad/include")
      set(glfw_dep "glfw")
    endif ()

    if (WIN32)
      set(IMGUI_LIB "lib/imgui.lib")
    else ()
      set(IMGUI_LIB "lib/libimgui.a")
    endif ()

    add_external_project(imgui STATIC
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG 085cff2fe58077a4a0bf1f9e9284814769141801 # docking branch > version "1.82"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${IMGUI_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/imgui/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt"
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/imgui/imgui_sw.cpp"
        "<SOURCE_DIR>/imgui_sw.cpp"
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/imgui/imgui_sw.h"
        "<SOURCE_DIR>/imgui_sw.h"
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/imgui/imgui_impl_generic.cpp"
        "<SOURCE_DIR>/backends/imgui_impl_generic.cpp"
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/imgui/imgui_impl_generic.h"
        "<SOURCE_DIR>/backends/imgui_impl_generic.h"
      DEPENDS
        ${glfw_dep}
      CMAKE_ARGS
        -DGLAD_INCLUDE_DIR:PATH=${glad_include_dir}
        -DGLFW_INCLUDE_DIR:PATH=${glfw_include_dir}
        -DENABLE_GL=${ENABLE_GL})

    add_external_library(imgui
      LIBRARY ${IMGUI_LIB})

  # imguizmoquat
  elseif (NAME STREQUAL "imguizmoquat")
    if (TARGET imguizmoquat)
      return()
    endif ()

    require_external(imgui)
    external_get_property(imgui INSTALL_DIR)

    if (WIN32)
      set(IMGUIZMOQUAT_LIB "lib/imguizmoquat.lib")
    else ()
      set(IMGUIZMOQUAT_LIB "lib/libimguizmoquat.a")
    endif ()

    add_external_project(imguizmoquat STATIC
      GIT_REPOSITORY https://github.com/braunms/imGuIZMO.quat.git
      GIT_TAG "v3.0a"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${IMGUIZMOQUAT_LIB}"
      DEPENDS imgui
      CMAKE_ARGS
        -DIMGUI_INCLUDE_DIR:PATH=${INSTALL_DIR}/include
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/imguizmoquat/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    add_external_library(imguizmoquat
      LIBRARY ${IMGUIZMOQUAT_LIB})

  # implot
  elseif (NAME STREQUAL "implot")
    if (TARGET implot)
      return()
    endif ()

    require_external(imgui)

    if (WIN32)
      set(IMPLOT_LIB "lib/implot.lib")
    else ()
      set(IMPLOT_LIB "lib/libimplot.a")
    endif ()

    if (WIN32)
      set(IMGUI_LIB "lib/imgui.lib")
    else ()
      set(IMGUI_LIB "lib/libimgui.a")
    endif ()

    external_get_property(imgui INSTALL_DIR)

    add_external_project(implot STATIC
      GIT_REPOSITORY https://github.com/epezent/implot.git
      GIT_TAG "v0.11"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${IMPLOT_LIB}"
      DEPENDS imgui
      CMAKE_ARGS
        -DIMGUI_LIBRARY:PATH=${INSTALL_DIR}/${IMGUI_LIB}
        -DIMGUI_INCLUDE_DIR:PATH=${INSTALL_DIR}/include
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/externals/implot/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    add_external_library(implot
      LIBRARY ${IMPLOT_LIB})

    external_get_property(implot SOURCE_DIR)
    target_include_directories(implot INTERFACE "${SOURCE_DIR}")

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

  # libpng
  elseif (NAME STREQUAL "libpng")
    if (TARGET libpng)
      return()
    endif ()

    require_external(zlib)

    if (MSVC)
      set(LIBPNG_LIB "lib/libpng16_static<SUFFIX>.lib")
    else ()
      set(LIBPNG_LIB "${CMAKE_INSTALL_LIBDIR}/libpng16<SUFFIX>.a")
    endif ()

    if (MSVC)
      set(ZLIB_LIB "lib/zlibstatic$<$<CONFIG:Debug>:d>.lib")
    else ()
      set(ZLIB_LIB "lib/libz.a")
    endif ()

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
  elseif (NAME STREQUAL "libzmq" OR NAME STREQUAL "libcppzmq")
    if (TARGET libzmq OR TARGET libcppzmq)
      return()
    endif ()

    set(ZMQ_VER "4_3_3")
    string(REPLACE "_" "." ZMQ_TAG "v${ZMQ_VER}")

    if (MSVC_IDE)
      set(MSVC_TOOLSET "-${CMAKE_VS_PLATFORM_TOOLSET}")
    else ()
      set(MSVC_TOOLSET "")
    endif ()

    if (WIN32)
      set(ZMQ_LIB "${CMAKE_INSTALL_LIBDIR}/libzmq${MSVC_TOOLSET}-mt-s<SUFFIX>-${ZMQ_VER}.lib")
    else ()
      set(ZMQ_LIB "${CMAKE_INSTALL_LIBDIR}/libzmq.a")
    endif ()

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
    if (WIN32)
      target_link_libraries(libzmq INTERFACE ws2_32 iphlpapi)
    endif ()

    add_external_headeronly_project(libcppzmq
      DEPENDS libzmq
      GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
      GIT_TAG "v4.6.0")

  # lua
  elseif (NAME STREQUAL "lua")
    if (TARGET lua)
      return()
    endif ()

    if (WIN32)
      set(LUA_LIB "lib/lua.lib")
    else ()
      set(LUA_LIB "lib/liblua.a")
    endif ()

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

  # megamol-shader-factory
  elseif (NAME STREQUAL "megamol-shader-factory")
    if (TARGET megamol-shader-factory)
      return()
    endif ()

    require_external(glad)

    if (WIN32)
      set(MEGAMOL_SHADER_FACTORY_LIB "lib/msf_combined.lib")
    else ()
      set(MEGAMOL_SHADER_FACTORY_LIB "lib/libmsf_combined.a")
    endif ()

    add_external_project(megamol-shader-factory STATIC
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/megamol-shader-factory.git
      GIT_TAG "v0.5"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${MEGAMOL_SHADER_FACTORY_LIB}"
      DEPENDS glad)

    add_external_library(megamol-shader-factory
      LIBRARY ${MEGAMOL_SHADER_FACTORY_LIB}
      INTERFACE_LIBRARIES glad)
    if (UNIX)
      target_link_libraries(megamol-shader-factory INTERFACE "stdc++fs")
    endif ()
    target_compile_definitions(megamol-shader-factory INTERFACE MSF_OPENGL_INCLUDE_GLAD2)

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

  # spdlog
  elseif (NAME STREQUAL "spdlog")
    if (TARGET spdlog)
      return()
    endif ()

    require_external(fmt)

    if (WIN32)
      set(SPDLOG_LIB "lib/spdlog<SUFFIX>.lib")
    else ()
      set(SPDLOG_LIB "${CMAKE_INSTALL_LIBDIR}/libspdlog<SUFFIX>.a")
    endif ()

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
        -Dfmt_DIR=${BINARY_DIR})

    add_external_library(spdlog
      LIBRARY ${SPDLOG_LIB}
      DEBUG_SUFFIX "d"
      DEPENDS fmt)

    target_compile_definitions(spdlog INTERFACE SPDLOG_FMT_EXTERNAL;SPDLOG_COMPILED_LIB)

  # tbb
  elseif (NAME STREQUAL "tbb")
    if (TARGET tbb)
      return()
    endif ()

    if (MSVC)
      set(EXTERNAL_EXE_LINKER_FLAGS "${EXTERNAL_EXE_LINKER_FLAGS} /NODEFAULTLIB:tbb12_debug.lib" CACHE STRING "" FORCE)
    endif ()

    if (WIN32)
      set(TBB_LIB "bin/tbb12.dll")
      set(TBB_LIB_IMPORT "${CMAKE_INSTALL_LIBDIR}/tbb12.lib")
    else ()
      set(TBB_LIB "${CMAKE_INSTALL_LIBDIR}/libtbb.so.12")
    endif ()

    add_external_project(tbb SHARED
      GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
      GIT_TAG v2021.3.0
      CMAKE_ARGS
        -DTBB_TEST=OFF
      BUILD_BYPRODUCTS
        "<INSTALL_DIR>/${TBB_LIB}"
        "<INSTALL_DIR>/${TBB_LIB_IMPORT}")

    add_external_library(tbb
      PROJECT tbb
      IMPORT_LIBRARY "${TBB_LIB_IMPORT}"
      LIBRARY "${TBB_LIB}")

  # tinyobjloader
  elseif (NAME STREQUAL "tinyobjloader")
    if (TARGET tinyobjloader)
      return()
    endif ()

    if (WIN32)
      set(TINYOBJLOADER_LIB "lib/tinyobjloader.lib")
    else ()
      set(TINYOBJLOADER_LIB "${CMAKE_INSTALL_LIBDIR}/libtinyobjloader.a")
    endif ()

    add_external_project(tinyobjloader STATIC
      GIT_REPOSITORY https://github.com/syoyo/tinyobjloader.git
      GIT_TAG "v2.0.0-rc1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TINYOBJLOADER_LIB}")

    add_external_library(tinyobjloader
      LIBRARY ${TINYOBJLOADER_LIB})

  # tinyply
  elseif (NAME STREQUAL "tinyply")
    if (TARGET tinyply)
      return()
    endif ()

    if (WIN32)
      set(TNY_LIB "${CMAKE_INSTALL_LIBDIR}/tinyply<SUFFIX>.lib")
    else ()
      set(TNY_LIB "${CMAKE_INSTALL_LIBDIR}/libtinyply<SUFFIX>.a")
    endif ()

    add_external_project(tinyply STATIC
      GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
      GIT_TAG "2.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TNY_LIB}"
      DEBUG_SUFFIX d
      CMAKE_ARGS
        -DSHARED_LIB=OFF)

    add_external_library(tinyply
      LIBRARY ${TNY_LIB}
      DEBUG_SUFFIX d)

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

  # vtkm
  elseif (NAME STREQUAL "vtkm")
    if (TARGET vtkm)
      return()
    endif ()

    set(VTKM_VER 1.4)
    set(LIB_VER 1)

    if (WIN32)
      set(VTKM_CONT_LIB "lib/vtkm_cont-${VTKM_VER}.lib")
      set(VTKM_RENDERER_LIB "lib/vtkm_rendering-${VTKM_VER}.lib")
      set(VTKM_WORKLET_LIB "lib/vtkm_worklet-${VTKM_VER}.lib")
    else ()
      set(VTKM_CONT_LIB "${CMAKE_INSTALL_LIBDIR}/libvtkm_cont-${VTKM_VER}.a")
      set(VTKM_RENDERER_LIB "${CMAKE_INSTALL_LIBDIR}/libvtkm_rendering-${VTKM_VER}.a")
      set(VTKM_WORKLET_LIB "${CMAKE_INSTALL_LIBDIR}/libvtkm_worklet-${VTKM_VER}.a")
    endif ()

    add_external_project(vtkm STATIC
      GIT_REPOSITORY https://gitlab.kitware.com/vtk/vtk-m.git
      GIT_TAG "v${VTKM_VER}.0"
      BUILD_BYPRODUCTS
        "<INSTALL_DIR>/${VTKM_CONT_LIB}"
        "<INSTALL_DIR>/${VTKM_RENDERER_LIB}"
        "<INSTALL_DIR>/${VTKM_WORKLET_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DBUILD_TESTING:BOOL=OFF
        -DVTKm_ENABLE_CUDA:BOOL=${vtkm_ENABLE_CUDA}
        -DVTKm_ENABLE_TESTING:BOOL=OFF
        -DVTKm_ENABLE_DEVELOPER_FLAGS:BOOL=OFF
        -DVTKm_ENABLE_EXAMPLES:BOOL=OFF
        -DVTKm_INSTALL_ONLY_LIBRARIES:BOOL=ON
        -DVTKm_USE_64BIT_IDS:BOOL=OFF
        #-DCMAKE_BUILD_TYPE=Release
      )

    add_external_library(vtkm
      PROJECT vtkm
      LIBRARY ${VTKM_CONT_LIB})

    add_external_library(vtkm_renderer
      PROJECT vtkm
      LIBRARY ${VTKM_RENDERER_LIB})

    add_external_library(vtkm_worklet
      PROJECT vtkm
      LIBRARY ${VTKM_WORKLET_LIB})

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

  # zlib
  elseif (NAME STREQUAL "zlib")
    if (TARGET zlib)
      return()
    endif ()

    if (MSVC)
      set(ZLIB_LIB "lib/zlibstatic<SUFFIX>.lib")
    else ()
      set(ZLIB_LIB "lib/libz.a")
    endif ()

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
