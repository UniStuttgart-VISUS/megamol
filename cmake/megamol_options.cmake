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
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(MEGAMOL_COMPILER_CLANG TRUE)
  include(clang)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(MEGAMOL_COMPILER_MSVC TRUE)
  include(msvc)
else()
  message(FATAL_ERROR
    "Unsupported compiler specified: '${CMAKE_CXX_COMPILER_ID}'")
endif()

option(USE_MPI "enable MPI in build" OFF)

set(MPI_GUESS_LIBRARY_NAME "undef" CACHE STRING "override MPI library, if necessary. ex: MSMPI, MPICH2")
if(USE_MPI)
  if(MPI_GUESS_LIBRARY_NAME STREQUAL "undef")
    message(FATAL_ERROR "you must set MPI_GUESS_LIBRARY_NAME to ovveride automatic finding of unwanted MPI libraries (or empty for default)")
  endif()
  find_package(MPI REQUIRED)
endif()

if(MPI_C_FOUND)
  # THIS IS THE APEX OF SHIT AND MUST DIE
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_C_COMPILE_FLAGS}")
  include_directories(${MPI_C_INCLUDE_PATH})
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}")
  set(LIBS ${LIBS} ${MPI_C_LIBRARIES})
  add_definitions(-DWITH_MPI -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
endif()

option(MEGAMOL_INSTALL_DEPENDENCIES "MegaMol dependencies in install" ON)
mark_as_advanced(MEGAMOL_INSTALL_DEPENDENCIES)

# Global Packages
set(CMAKE_THREAD_PREFER_PTHREAD)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
