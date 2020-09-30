# C++ standard
set(CMAKE_CXX_STANDARD 17)

# Word size
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITS 64)
else()
  set(BITS 32)
endif()

# Build types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})

# Compiler flags (inspired by OSPRay build)
option(DISABLE_WARNINGS "Disables all compiler warnings" ON)
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

# Install
option(MEGAMOL_INSTALL_DEPENDENCIES "MegaMol dependencies in install" ON)
mark_as_advanced(MEGAMOL_INSTALL_DEPENDENCIES)

# CUDA
option(ENABLE_CUDA "Enable CUDA, which is needed for certain plugins" OFF)
if(ENABLE_CUDA)
  enable_language(CUDA)
endif()

# GLFW
option(USE_GLFW "Use GLFW" ON)

# MPI
option(ENABLE_MPI "Enable MPI support" OFF)
set(MPI_GUESS_LIBRARY_NAME "undef" CACHE STRING "Override MPI library name, e.g., MSMPI, MPICH2")
if(ENABLE_MPI)
  if(MPI_GUESS_LIBRARY_NAME STREQUAL "undef")
    message(FATAL_ERROR "you must set MPI_GUESS_LIBRARY_NAME to ovveride automatic finding of unwanted MPI libraries (or empty for default)")
  endif()
  find_package(MPI REQUIRED)
  if(MPI_C_FOUND)
    target_compile_definitions(MPI::MPI_C INTERFACE "-DWITH_MPI")
endif()
endif()

# Threading (XXX: this is a bit wonky due to Ubuntu/clang)
include(CheckFunctionExists)
check_function_exists(pthread_create HAVE_PTHREAD)
if(HAVE_PTHREAD)
  set(CMAKE_THREAD_PREFER_PTHREAD ON)
  find_package(Threads REQUIRED)
endif()

# OpenMP
if(UNIX)
  find_package(OpenMP)
endif()
if(OPENMP_FOUND OR WIN32)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OpenGL
find_package(OpenGL REQUIRED)
