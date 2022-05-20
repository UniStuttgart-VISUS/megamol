# MegaMol
# Copyright (c) 2020, MegaMol Dev Team
# All rights reserved.
#
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# Policies
cmake_policy(SET CMP0054 NEW) # if arguments
cmake_policy(SET CMP0079 NEW) # use targets from other directories
cmake_policy(SET CMP0091 NEW) # MSVC runtime
if (POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW) # CUDA architectures
endif ()

# Disable in source build
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Project
project(megamol
  VERSION 1.3.0
  LANGUAGES C CXX)

# Default build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()

# Default install prefix
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE)
endif ()
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "default install path" FORCE) # This will replace "\" by "/"

# Modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# MegaMol config
include(megamol_config)

# MegaMol build info library (version + config)
include(megamol_build_info)

# Externals
include("externals/CMakeExternals.cmake")

# MegaMol targets

# Frontend Resources, Input Events interfaces
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend/resources)

# Frontend Services (e.g. OpenGL_GLFW provider)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend/services)

# Vislib
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vislib)
if (ENABLE_GL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vislib_gl)
endif ()

# Core
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)
if (ENABLE_GL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core_gl)
endif ()

# MegaMol.exe
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend/main)

# Remoteconsole
add_subdirectory(remoteconsole)

# Plugins
add_subdirectory(plugins)
if (BUILD_FRONTEND)
  if (ENABLE_GL)
    target_link_libraries(plugins INTERFACE core_gl)
    target_sources(plugins INTERFACE $<TARGET_OBJECTS:core_gl>)
  endif ()
  target_link_libraries(megamol PRIVATE plugins)
endif ()

# Utils
add_subdirectory(utils)

# Add directory structure for visual studio
if (WIN32)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD TRUE)
endif ()

# Examples
option(EXAMPLES "Get the MegaMol example repository" ON)
if (EXAMPLES)
  set(EXAMPLES_DIR "${CMAKE_BINARY_DIR}/examples" CACHE PATH "Download directory of the examples")
  option(EXAMPLES_UPDATE "Pull updates from the examples repo" ON)
  if (NOT EXISTS "${EXAMPLES_DIR}")
    message(STATUS "Downloading examples")
    execute_process(COMMAND
      ${GIT_EXECUTABLE} clone https://github.com/UniStuttgart-VISUS/megamol-examples.git "${EXAMPLES_DIR}" --depth 1
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  elseif (EXAMPLES_UPDATE)
    message(STATUS "Pull example updates")
    execute_process(COMMAND
      ${GIT_EXECUTABLE} pull
      WORKING_DIRECTORY "${EXAMPLES_DIR}"
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif ()
  option(EXAMPLES_INSTALL "Install examples" ON)
  if (EXAMPLES_INSTALL)
    install(DIRECTORY "${EXAMPLES_DIR}/" DESTINATION "${CMAKE_INSTALL_PREFIX}/examples" PATTERN ".git" EXCLUDE)
  endif ()
endif ()

# Tests
option(TESTS "Get the MegaMol tests repository" OFF)
if (TESTS)
  set(TESTS_DIR "${CMAKE_BINARY_DIR}/tests" CACHE PATH "Download directory of the tests")
  option(TESTS_UPDATE "Pull updates from the tests repo" ON)
  if (NOT EXISTS "${TESTS_DIR}")
    message(STATUS "Downloading tests")
    execute_process(COMMAND
      ${GIT_EXECUTABLE} clone https://github.com/UniStuttgart-VISUS/megamol-tests.git "${TESTS_DIR}" --depth 1
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  elseif (TESTS_UPDATE)
    message(STATUS "Pull tests updates")
    execute_process(COMMAND
      ${GIT_EXECUTABLE} pull
      WORKING_DIRECTORY "${TESTS_DIR}"
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif ()
  install(DIRECTORY "${TESTS_DIR}/" DESTINATION "${CMAKE_INSTALL_PREFIX}/tests" PATTERN ".git" EXCLUDE)
endif ()
