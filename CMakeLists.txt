#
# MegaMol™ Core
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

project(megamol)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}
    CACHE PATH "default install path" FORCE)
endif()

# This will replace "\" by "/"
set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH "default install path" FORCE)

# Directories
set(MEGAMOL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MEGAMOL_VISLIB_DIR ${MEGAMOL_DIR}/vislib)
set(MEGAMOL_CORE_DIR ${MEGAMOL_DIR}/core)
set(MEGAMOL_CONSOLE_DIR ${MEGAMOL_DIR}/console)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Versions
include(global_versioninfo)

# MegaMol options
include(megamol_options)

# MegaMol macros
include(check_mmdep)

# Clang-format
include(ClangFormat)

# External
include("CMakeExternals.cmake")

# Vislib
add_subdirectory(${MEGAMOL_VISLIB_DIR})

# Core
add_subdirectory(${MEGAMOL_CORE_DIR})

# Console
set(CONSOLE_NAME mmconsole)
add_subdirectory(${MEGAMOL_CONSOLE_DIR})

# Add directory structure for visual studio
if(WIN32)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD TRUE)
endif()

# Configurator
if(WIN32)
  # NuGet
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/NuGet.Config
    ${CMAKE_CURRENT_BINARY_DIR}/NuGet.Config COPYONLY)
  add_subdirectory(configurator)
endif()

# Plugins
add_subdirectory(plugins)

# Utils
add_subdirectory(utils)

# Examples
option(EXAMPLES "Get the MegaMol example repository" ON)
if(EXAMPLES)
  if(NOT EXISTS "${CMAKE_BINARY_DIR}/examples")
    message(STATUS "Downloading examples")
    execute_process(COMMAND
      ${GIT_EXECUTABLE} clone https://github.com/UniStuttgart-VISUS/megamol-examples.git examples --depth 1
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  install(DIRECTORY "${CMAKE_BINARY_DIR}/examples" DESTINATION "${CMAKE_INSTALL_PREFIX}" PATTERN ".git" EXCLUDE)
endif()

# Validate dependencies
foreach(d ${DEP_LIST})
  string(REGEX REPLACE " +" ";" SINGLE_LIST "${d}")
  list(GET SINGLE_LIST 0 plugin)
  list(REMOVE_AT SINGLE_LIST 0)
  check_mmdep(${plugin} "${SINGLE_LIST}")
endforeach()
unset(DEP_LIST CACHE)
