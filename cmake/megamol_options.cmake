# Word size detection
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITS 64)
else()
  set(BITS 32)
endif()

set(CMAKE_CXX_STANDARD 17)

if(UNIX)
    find_package(OpenMP)
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})

# Compiler flags (inspired by OSPRay build)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(MEGAMOL_COMPILER_ICC TRUE)
    include(icc)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(MEGAMOL_COMPILER_GCC TRUE)
    include(gcc)
# TODO: clang is unsupported for now
#elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#    set(MEGAMOL_COMPILER_CLANG TRUE)
#    include(clang)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(MEGAMOL_COMPILER_MSVC TRUE)
    include(msvc)
else()
    message(FATAL_ERROR
        "Unsupported compiler specified: '${CMAKE_CXX_COMPILER_ID}'")
endif()

option(MEGAMOL_INSTALL_DEPENDENCIES "MegaMol dependencies in install" ON)
mark_as_advanced(MEGAMOL_INSTALL_DEPENDENCIES)

# Global Packages
set(CMAKE_THREAD_PREFER_PTHREAD)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND OR MSVC)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
