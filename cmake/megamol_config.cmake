# MegaMol configuration

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
# Note: special C++ and C-Compiler flags should be set for each language separately as done below.
# Otherwise, a possible compilation with CUDA will propagate those flags to the CUDA-Compiler and lead to a crash.
# For certain systems, those flags should be set for both C++ and C compilers to work properly.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # nothing to do
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # TODO git history suggests this was required for cuda in 2019, still required?
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fsized-deallocation>)

  # Prevent build fail.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wno-narrowing>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wno-non-pod-varargs>)
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

# Clang-tidy
# Configure with:
# > cmake -DMEGAMOL_RUN_CLANG_TIDY=ON -DMEGAMOL_WARNING_LEVEL=Off ../megamol
# Build NOT in parallel, otherwise clang-tidy will mess up files!
#
# Alternative:
# > cd build && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../megamol
# > cd ../megamol && run-clang-tidy-14 -p ../build -fix -j 32
# This uses the run-clang-tidy wrapper allowing very fast parallel execution, but also tries to fix not existent source files in the binary dir.
#
# Both solutions do not exclude 3rd directories and clang-tidy results seems to sometime trigger edge cases with weird results.
# Therefore, results should be committed manually and currently do not run clang-tidy within the CI pipeline.
option(MEGAMOL_RUN_CLANG_TIDY "Run clang-tidy." OFF)
mark_as_advanced(FORCE MEGAMOL_RUN_CLANG_TIDY)
if (MEGAMOL_RUN_CLANG_TIDY)
  SET(CMAKE_CXX_CLANG_TIDY "clang-tidy-14;--fix")
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

# imgui
# Set IMGUI_USER_CONFIG globally on imgui target for all users.
find_package(imgui CONFIG REQUIRED)
target_compile_definitions(imgui::imgui INTERFACE "IMGUI_USER_CONFIG=\"${CMAKE_CURRENT_SOURCE_DIR}/cmake/imgui/megamol_imgui_config.h\"")
