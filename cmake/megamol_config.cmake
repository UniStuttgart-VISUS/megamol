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
# Note: special C++ and C-Compiler flags should be set for each language seperately as done below.
# Otherwise, a possible compilation with CUDA will propagate those flags to the CUDA-Compiler and lead to a crash.
# For certain systems, those flags should be set for both C++ and C compilers to work properly
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # nothing to do
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # TODO git history suggests this was required for cuda in 2019, still required?
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fsized-deallocation>)

  # Prevent build fail.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wno-narrowing>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wno-non-pod-vararg>)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/permissive->)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/Zc:twoPhase->)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
  add_compile_definitions("NOMINMAX")
else ()
  message(FATAL_ERROR "Unsupported compiler specified: '${CMAKE_CXX_COMPILER_ID}'")
endif ()

# Set RPath to "../lib" on binary install
if (UNIX)
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif ()

# Dependencies

# OpenMP
find_package(OpenMP REQUIRED)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# OpenGL
if (MEGAMOL_USE_OPENGL)
  find_package(OpenGL REQUIRED)
endif ()

# CUDA
if (MEGAMOL_USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES FALSE)
endif ()

# MPI
if (MEGAMOL_USE_MPI)
  find_package(MPI REQUIRED)
endif ()

# CGAL
if (MEGAMOL_USE_CGAL)
  find_package(CGAL REQUIRED)
  set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_MINSIZEREL Release)
  set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
endif ()
