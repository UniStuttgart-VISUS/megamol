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
      set(ADIOS2_LIB "${CMAKE_INSTALL_LIBDIR}/libadios2.so")
    endif()

    add_external_project(adios2
      GIT_REPOSITORY https://github.com/ornladios/ADIOS2.git
      GIT_TAG "v2.3.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ADIOS2_LIB}"
      CMAKE_ARGS 
        -DBUILD_TESTING=OFF
        -DADIOS2_USE_BZip2=OFF 
        -DADIOS2_USE_Fortran=OFF
        -DADIOS2_USE_HDF5=OFF 
        -DADIOS2_USE_Python=OFF
        -DADIOS2_USE_SST=OFF 
        -DADIOS2_USE_SZ=OFF
        -DADIOS2_USE_SysVShMem=OFF 
        -DADIOS2_USE_ZFP=OFF
        -DADIOS2_USE_ZeroMQ=OFF 
        -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

    add_external_library(adios2
      IMPORT_LIBRARY ${ADIOS2_IMPORT_LIB}
      LIBRARY ${ADIOS2_LIB})

  # bhtsne
  elseif(NAME STREQUAL "bhtsne")
    if(TARGET bhtsne)
      return()
    endif()

    if(WIN32)
      set(BHTSNE_IMPORT_LIB "lib/bhtsne.lib")
      set(BHTSNE_LIB "bin/bhtsne.dll")
    else()
      set(BHTSNE_LIB "lib/libbhtsne.so")
    endif()

    add_external_project(bhtsne
      GIT_REPOSITORY https://github.com/lvdmaaten/bhtsne.git
      GIT_TAG "36b169c88250d0afe51828448dfdeeaa508f13bc"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${BHTSNE_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/bhtsne/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt"
      CMAKE_ARGS
        -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE)

    add_external_library(bhtsne
      IMPORT_LIBRARY ${BHTSNE_IMPORT_LIB}
      LIBRARY ${BHTSNE_LIB})

  # glfw3
  elseif(NAME STREQUAL "glfw3")
    if(TARGET glfw3)
      return()
    endif()

    if(WIN32)
      set(GLFW_IMPORT_LIB "lib/glfw3dll.lib")
      set(GLFW_LIB "lib/glfw3.dll")
      set(GLFW_COPY COPY "\"<INSTALL_DIR>/lib/glfw3.dll\" \"<INSTALL_DIR>/bin/glfw3.dll\"")
    else()
      set(GLFW_LIB "lib/libglfw.so")
    endif()

    add_external_project(glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG "3.2.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${GLFW_LIB}"
      ${GLFW_COPY}
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DGLFW_BUILD_EXAMPLES=OFF
        -DGLFW_BUILD_TESTS=OFF
        -DGLFW_BUILD_DOCS=OFF)

    add_external_library(glfw3
      PROJECT glfw
      IMPORT_LIBRARY ${GLFW_IMPORT_LIB}
      LIBRARY ${GLFW_LIB})

  # IceT
  elseif(NAME STREQUAL "IceT")
    if(TARGET IceTCore)
      return()
    endif()

    if(WIN32)
      set(ICET_CORE_IMPORT_LIB "lib/IceTCore.lib")
      set(ICET_GL_IMPORT_LIB "lib/IceTGL.lib")
      set(ICET_MPI_IMPORT_LIB "lib/IceTMPI.lib")
      set(ICET_CORE_LIB "bin/IceTCore.dll")
      set(ICET_GL_LIB "bin/IceTGL.dll")
      set(ICET_MPI_LIB "bin/IceTMPI.dll")
    else()
      include(GNUInstallDirs)
      set(ICET_CORE_LIB "lib/libIceTCore.so")
      set(ICET_GL_LIB "lib/libIceTGL.so")
      set(ICET_MPI_LIB "lib/libIceTMPI.so")
    endif()
    
    add_external_project(IceT
      GIT_REPOSITORY https://gitlab.kitware.com/icet/icet.git
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ICET_CORE_LIB}" "<INSTALL_DIR>/${ICET_GL_LIB}" "<INSTALL_DIR>/${ICET_MPI_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DICET_BUILD_TESTING=OFF
        -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

    add_external_library(IceTCore
      PROJECT IceT
      IMPORT_LIBRARY ${ICET_CORE_IMPORT_LIB}
      LIBRARY ${ICET_CORE_LIB})

    add_external_library(IceTGL
      PROJECT IceT
      IMPORT_LIBRARY ${ICET_GL_IMPORT_LIB}
      LIBRARY ${ICET_GL_LIB})

    add_external_library(IceTMPI
      PROJECT IceT
      IMPORT_LIBRARY ${ICET_MPI_IMPORT_LIB}
      LIBRARY ${ICET_MPI_LIB})

  # imgui
  elseif(NAME STREQUAL "imgui")
    if(TARGET imgui)
      return()
    endif()

    if(WIN32)
      set(IMGUI_IMPORT_LIB "lib/imgui.lib")
      set(IMGUI_LIB "bin/imgui.dll")
    else()
      set(IMGUI_LIB "lib/libimgui.so")
    endif()

    add_external_project(imgui
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG "v1.70"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${IMGUI_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/imgui/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    external_get_property(imgui SOURCE_DIR)

    add_external_library(imgui
      IMPORT_LIBRARY ${IMGUI_IMPORT_LIB}
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
      set(LIBPNG_IMPORT_LIB "lib/libpng16.lib")
      set(LIBPNG_LIB "bin/libpng16.dll")
    else()
      include(GNUInstallDirs)
      set(LIBPNG_LIB "${CMAKE_INSTALL_LIBDIR}/libpng16.so")
    endif()

    if(MSVC)
      set(ZLIB_LIB "lib/zlib.lib")
    else()
      include(GNUInstallDirs)
      set(ZLIB_LIB "lib/libz.so")
    endif()

    external_get_property(zlib INSTALL_DIR)

    add_external_project(libpng
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/libpng.git
      GIT_TAG "v1.6.34"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${LIBPNG_LIB}"
      DEPENDS zlib_ext
      CMAKE_ARGS
        -DPNG_BUILD_ZLIB=ON
        -DPNG_SHARED=ON
        -DPNG_STATIC=OFF
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
      IMPORT_LIBRARY ${LIBPNG_IMPORT_LIB}
      LIBRARY ${LIBPNG_LIB}
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
      set(ZMQ_IMPORT_LIB "lib/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.lib")
      set(ZMQ_LIB "bin/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.dll")
    else()
      include(GNUInstallDirs)
      set(ZMQ_LIB "${CMAKE_INSTALL_LIBDIR}/libzmq.so")
    endif()

    add_external_project(libzmq
      GIT_REPOSITORY https://github.com/zeromq/libzmq.git
      GIT_TAG 56ace6d03f521b9abb5a50176ec7763c1b77afa9
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZMQ_LIB}"
      CMAKE_ARGS
        -DZMQ_BUILD_TESTS=OFF
        -DENABLE_PRECOMPILED=OFF)

    add_external_library(libzmq
      IMPORT_LIBRARY ${ZMQ_IMPORT_LIB}
      LIBRARY ${ZMQ_LIB})

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
    else()
      set(QUICKHULL_LIB "lib/libquickhull.so")
    endif()

    add_external_project(quickhull
      GIT_REPOSITORY https://github.com/akuukka/quickhull.git
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${QUICKHULL_LIB}"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/quickhull/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt"
      CMAKE_ARGS
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC)

    add_external_library(quickhull
      IMPORT_LIBRARY ${QUICKHULL_IMPORT_LIB}
      LIBRARY ${QUICKHULL_LIB})

  # snappy
  elseif(NAME STREQUAL "snappy")
    if(TARGET snappy)
      return()
    endif()

    if(WIN32)
      set(SNAPPY_IMPORT_LIB "lib/snappy.lib")
      set(SNAPPY_LIB "bin/snappy.dll")
    else()
      include(GNUInstallDirs)
      set(SNAPPY_LIB "${CMAKE_INSTALL_LIBDIR}/libsnappy.so")
    endif()

    add_external_project(snappy
      GIT_REPOSITORY https://github.com/google/snappy.git
      GIT_TAG "1.1.7"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${SNAPPY_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DSNAPPY_BUILD_TESTS=OFF
        -DCMAKE_BUILD_TYPE=Release)

    add_external_library(snappy
      IMPORT_LIBRARY ${SNAPPY_IMPORT_LIB}
      LIBRARY ${SNAPPY_LIB})

  # tinyobjloader
  elseif(NAME STREQUAL "tinyobjloader")
    if(TARGET tinyobjloader)
      return()
    endif()

    if(WIN32)
      set(TINYOBJLOADER_IMPORT_LIB "lib/tinyobjloader.lib")
      set(TINYOBJLOADER_LIB "bin/tinyobjloader.dll")
    else()
      include(GNUInstallDirs)
      set(TINYOBJLOADER_LIB "${CMAKE_INSTALL_LIBDIR}/libtinyobjloader.so")
    endif()

    add_external_project(tinyobjloader
      GIT_REPOSITORY https://github.com/syoyo/tinyobjloader.git
      GIT_TAG "v2.0.0-rc1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TINYOBJLOADER_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE
        -DCMAKE_C_FLAGS=-fPIC
        -DCMAKE_CXX_FLAGS=-fPIC)

    add_external_library(tinyobjloader
      IMPORT_LIBRARY ${TINYOBJLOADER_IMPORT_LIB}
      LIBRARY ${TINYOBJLOADER_LIB})

  # tinyply
  elseif(NAME STREQUAL "tinyply")
    if(TARGET tinyply)
      return()
    endif()

    if(WIN32)
      set(TNY_IMPORT_LIB "lib/tinyply.lib")
      set(TNY_LIB "bin/tinyply.dll")
    else()
      include(GNUInstallDirs)
      set(TNY_LIB "lib/libtinyply.so")
    endif()

    add_external_project(tinyply
      GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
      GIT_TAG "2.1"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${TNY_LIB}"
      CMAKE_ARGS
        -DSHARED_LIB=ON)

    add_external_library(tinyply
      IMPORT_LIBRARY ${TNY_IMPORT_LIB}
      LIBRARY ${TNY_LIB})

  # zfp
  elseif(NAME STREQUAL "zfp")
    if(TARGET zfp)
      return()
    endif()

    if(WIN32)
      set(ZFP_IMPORT_LIB "lib/zfp.lib")
      set(ZFP_LIB "bin/zfp.dll")
    else()
      include(GNUInstallDirs)
      set(ZFP_LIB "${CMAKE_INSTALL_LIBDIR}/libzfp.so")
    endif()

    add_external_project(zfp
      GIT_REPOSITORY https://github.com/LLNL/zfp.git
      GIT_TAG "0.5.2"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZFP_LIB}"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_UTILITIES=OFF
        -DBUILD_TESTING=OFF
        -DZFP_WITH_ALIGNED_ALLOC=ON
        -DZFP_WITH_CACHE_FAST_HASH=ON
        -DCMAKE_BUILD_TYPE=Release)

    add_external_library(zfp
      IMPORT_LIBRARY ${ZFP_IMPORT_LIB}
      LIBRARY ${ZFP_LIB})

  # zlib
  elseif(NAME STREQUAL "zlib")
    if(TARGET zlib)
      return()
    endif()

    if(MSVC)
      set(ZLIB_IMPORT_LIB "lib/zlib.lib")
      set(ZLIB_LIB "bin/zlib.dll")
    else()
      include(GNUInstallDirs)
      set(ZLIB_LIB "lib/libz.so")
    endif()

    add_external_project(zlib
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG "v1.2.11"
      BUILD_BYPRODUCTS "<INSTALL_DIR>/${ZLIB_LIB}"
      CMAKE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON)

    add_external_library(zlib
      IMPORT_LIBRARY ${ZLIB_IMPORT_LIB}
      LIBRARY ${ZLIB_LIB})

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
