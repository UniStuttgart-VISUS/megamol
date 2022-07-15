# MegaMol configuration

# CMake Modules
include(CMakeDependentOption)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

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

# Debug
# Note: 'NDEBUG' is set by default from CMake if not in debug config.
# TODO do we still need both or can we switch to one?
add_compile_definitions("$<$<CONFIG:DEBUG>:DEBUG>")
add_compile_definitions("$<$<CONFIG:DEBUG>:_DEBUG>")

# Compiler flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # nothing to do
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options("-fsized-deallocation") # TODO git history suggests this was required for cuda in 2019, still required?
  add_compile_options("-Wno-narrowing" "-Wno-non-pod-varargs") # Prevent build fail.
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options("/MP" "/permissive-" "/Zc:twoPhase-" "/utf-8")
  add_compile_definitions("NOMINMAX")
else ()
  message(FATAL_ERROR "Unsupported compiler specified: '${CMAKE_CXX_COMPILER_ID}'")
endif ()

# Set RPath to "../lib" on binary install
if (UNIX)
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib")
endif ()

# Dependencies

# OpenMP
find_package(OpenMP REQUIRED)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# OpenGL
if (MEGAMOL_USE_OPENGL)
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

# CGAL
if (MEGAMOL_USE_CGAL)
  add_compile_definitions(WITH_CGAL)
  find_package(CGAL REQUIRED)

  if (NOT TARGET CGAL::CGAL)
    message(FATAL_ERROR "Target for CGAL not found")
  endif ()

  if (TARGET CGAL)
    set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
  endif ()
endif()

# Profiling
cmake_dependent_option(ENABLE_PROFILING "Enable profiling code" OFF "MEGAMOL_USE_OPENGL" OFF)
if (ENABLE_PROFILING)
  add_compile_definitions(PROFILING)
endif ()

# VR Service / mwk-mint, interop, Spout2
cmake_dependent_option(ENABLE_VR_SERVICE_UNITY_KOLABBW "Enable KolabBW-Unity-Interop in VR Service" OFF "MEGAMOL_USE_OPENGL" OFF)
if (ENABLE_VR_SERVICE_UNITY_KOLABBW)
  add_compile_definitions(WITH_VR_SERVICE_UNITY_KOLABBW)
endif ()

# CUE
cmake_dependent_option(ENABLE_CUESDK "Enable CUE for highlighting hotkeys on Corsair Keyboards" OFF "WIN32" OFF)
if (ENABLE_CUESDK)
  add_compile_definitions(CUESDK_ENABLED)
endif ()
