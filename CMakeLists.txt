# MegaMol
# Copyright (c) 2020, MegaMol Dev Team
# All rights reserved.
#
cmake_minimum_required(VERSION 3.15...3.26 FATAL_ERROR)

# Vcpkg setup
set(MEGAMOL_VCPKG_VERSION "2023.06.20") # Update default-registry baseline in vcpkg-configuration.json when changing!

include(cmake/megamol_vcpkg_setup.cmake)
include(cmake/megamol_feature_option.cmake)
include(cmake/megamol_vcpkg_empty_port.cmake)

# MegaMol vcpkg features
megamol_feature_option(CGAL "Enable CGAL support." OFF)
megamol_feature_option(CUDA "Enable CUDA support." OFF)
megamol_feature_option(MPI "Enable MPI support." OFF)
megamol_feature_option(OPENGL "Enable OpenGL support." ON)
megamol_feature_option(OSPRAY "Enable OSPRay support." OFF)
megamol_feature_option(POWER "Enable power measurement support." OFF)
megamol_feature_option(PROFILING "Enable profiling support." OFF)
megamol_feature_option(STACKTRACE "Enable Boost.Stacktrace support." OFF)
megamol_feature_option(TRACY "Enable tracy support." OFF)
megamol_feature_option(VTKM "Enable VTK-m support." OFF)
# MegaMol vcpkg features (dependent options)
megamol_feature_option(CUESDK "Enable Corsair CUESDK support." OFF "WIN32")
megamol_feature_option(OPENGL_DEBUGGROUPS "Inject OpenGL debug groups into calls." OFF "MEGAMOL_USE_OPENGL")
megamol_feature_option(VR_INTEROP "Enable MegaMol-Unity VR Interop via Spout2." OFF "WIN32;MEGAMOL_USE_OPENGL")

# Disable in source build
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Catch deprecated options before project()
include(cmake/deprecation_warning.cmake)

# Project
project(megamol
  VERSION 1.3.0
  LANGUAGES C CXX)

# Allow CI dependency build to stop here.
if (MEGAMOL_STOP_AFTER_VCPKG)
  return()
endif ()

# Default build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()

# Default install prefix
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif ()
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install path prefix, prepended onto install directories." FORCE) # This will replace "\" by "/"

# Modules
include(GNUInstallDirs)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# MegaMol config
include(megamol_config)

# MegaMol build info library (version + config)
include(megamol_build_info)

# MegaMol targets

# Frontend Resources, Input Events interfaces
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend/resources)

# Frontend Services (e.g. OpenGL_GLFW provider)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend/services)

# Vislib
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vislib)
if (MEGAMOL_USE_OPENGL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vislib_gl)
endif ()

# Core
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)
if (MEGAMOL_USE_OPENGL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core_gl)
endif ()

# Plugins
add_subdirectory(plugins)
if (MEGAMOL_USE_OPENGL)
  target_link_libraries(plugins INTERFACE core_gl)
  target_sources(plugins INTERFACE $<TARGET_OBJECTS:core_gl>)
endif ()

# MegaMol.exe
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend/main)
target_link_libraries(megamol PRIVATE plugins)

# Utils
add_subdirectory(utils)

# Add directory structure for visual studio
if (WIN32)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD TRUE)
endif ()

# Additional downloaded and installed repositories
include(cmake/megamol_extra_repo.cmake)

# Examples
megamol_extra_repo(examples
  GIT_REPO "https://github.com/UniStuttgart-VISUS/megamol-examples.git"
  DEFAULT_ON)

# Tests
megamol_extra_repo(tests
  GIT_REPO "https://github.com/UniStuttgart-VISUS/megamol-tests.git")

# Install vcpkg libraries
# X_VCPKG_APPLOCAL_DEPS_INSTALL currently does only work for windows, in addition only actually used dll's are
# installed. This is a problem for libraries that load dll's dynamically on runtime (i.e. ospray).
# Therefore, we just copy all vcpkg libraries to our install dir, until vcpkg may has a better option in future.
if (WIN32)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-windows/$<$<CONFIG:Debug>:debug/>bin/
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    FILES_MATCHING PATTERN "*.dll" PATTERN "*.pdb")
else ()
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-linux/$<$<CONFIG:Debug>:debug/>lib/
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    FILES_MATCHING PATTERN "*.so*")
endif ()
