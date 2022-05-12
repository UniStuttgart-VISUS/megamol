# MegaMol configuration

# CMake Modules
include(CMakeDependentOption)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# Warnings
set(MEGAMOL_WARNING_LEVEL Default CACHE STRING "Define compiler warning level.")
set_property(CACHE MEGAMOL_WARNING_LEVEL PROPERTY STRINGS "Off" "Default" "All")
if ("${MEGAMOL_WARNING_LEVEL}" STREQUAL "Off")
  if (MSVC)
    add_compile_options("/W0")
  else ()
    add_compile_options("-w")
  endif ()
elseif ("${MEGAMOL_WARNING_LEVEL}" STREQUAL "All")
  if (MSVC)
    add_compile_options("/W4 /external:anglebrackets /external:W0")
  else ()
    add_compile_options("-Wall -Wextra -pedantic")
  endif ()
endif ()


# Compiler flags (inspired by OSPRay build)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
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

# Dependencies

# OpenMP
find_package(OpenMP REQUIRED)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# OpenGL
option(ENABLE_GL "Enable GL support" ON)
if (ENABLE_GL)
  add_compile_definitions(WITH_GL)
  find_package(OpenGL REQUIRED)
endif ()

# CUDA
option(ENABLE_CUDA "Enable CUDA support" OFF)
if (ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES FALSE)
endif ()

# MPI
option(ENABLE_MPI "Enable MPI support" OFF)
set(MPI_GUESS_LIBRARY_NAME "undef" CACHE STRING "Override MPI library name, e.g., MSMPI, MPICH2")
if (ENABLE_MPI)
  if (MPI_GUESS_LIBRARY_NAME STREQUAL "undef")
    message(FATAL_ERROR "you must set MPI_GUESS_LIBRARY_NAME to ovveride automatic finding of unwanted MPI libraries (or empty for default)")
  endif ()
  find_package(MPI REQUIRED)
  if (MPI_C_FOUND)
    target_compile_definitions(MPI::MPI_C INTERFACE "-DWITH_MPI")
  endif ()
endif ()

# Profiling
cmake_dependent_option(ENABLE_PROFILING "Enable profiling code" OFF "ENABLE_GL" OFF)
if (ENABLE_PROFILING)
  add_compile_definitions(PROFILING)
endif ()

# VR Service / mwk-mint, interop, Spout2
cmake_dependent_option(ENABLE_VR_SERVICE_UNITY_KOLABBW "Enable KolabBW-Unity-Interop in VR Service" OFF "ENABLE_GL" OFF)
if (ENABLE_VR_SERVICE_UNITY_KOLABBW)
  add_compile_definitions(WITH_VR_SERVICE_UNITY_KOLABBW)
endif ()

# CUE
cmake_dependent_option(ENABLE_CUESDK "Enable CUE for highlighting hotkeys on Corsair Keyboards" OFF "WIN32" OFF)
if (ENABLE_CUESDK)
  add_compile_definitions(CUESDK_ENABLED)
endif ()
