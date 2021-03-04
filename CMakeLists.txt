# MegaMol
# Copyright (c) 2020, MegaMol Dev Team
# All rights reserved.
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0079 NEW) # Needed for static build, requires CMake 3.13!

project(megamol)

# Default to install to source directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
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
include("externals/CMakeExternals.cmake")

# Frontend Resources, Input Events interfaces 
add_subdirectory(${MEGAMOL_DIR}/frontend/resources)

# Frontend Services (e.g. OpenGL_GLFW provider)
add_subdirectory(${MEGAMOL_DIR}/frontend/services)

# Vislib
add_subdirectory(${MEGAMOL_VISLIB_DIR})

# Core
add_subdirectory(${MEGAMOL_CORE_DIR})

# Console
set(CONSOLE_NAME mmconsole)
add_subdirectory(${MEGAMOL_CONSOLE_DIR})

# Remoteconsole
add_subdirectory(remoteconsole)

# MegaMol.exe, new frontend, Main3000
add_subdirectory(${MEGAMOL_DIR}/frontend/main)

# Add directory structure for visual studio
if(WIN32)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD TRUE)
endif()

# Plugin register function
# This function must be called by every plugin.
# Within this function we need to define every binary which should include plugins.
function(megamol_register_plugin TARGET)
  if (WIN32)
    if (BUILD_CONSOLE)
      target_link_libraries(mmconsole PRIVATE ${TARGET})
      target_link_options(mmconsole PRIVATE "/WHOLEARCHIVE:${TARGET}")
    endif()
    if (BUILD_FRONTEND)
      target_link_libraries(megamol PRIVATE ${TARGET})
      target_link_options(megamol PRIVATE "/WHOLEARCHIVE:${TARGET}")
    endif()
  else ()
    if (BUILD_CONSOLE)
      target_link_libraries(mmconsole PRIVATE
        -Wl,--whole-archive
        ${TARGET}
        -Wl,--no-whole-archive)
    endif()
    if (BUILD_FRONTEND)
      target_link_libraries(megamol PRIVATE
        -Wl,--whole-archive
        ${TARGET}
        -Wl,--no-whole-archive)
    endif()
  endif ()
endfunction(megamol_register_plugin)

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
