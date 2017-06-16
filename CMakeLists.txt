cmake_minimum_required(VERSION 3.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(megamol)

#DEBUG
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
#DEBUG

# configure directories
set(MEGAMOL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MEGAMOL_VISLIB_DIR ${MEGAMOL_DIR}/vislib)
set(MEGAMOL_CORE_DIR ${MEGAMOL_DIR}/core)
set(MEGAMOL_CONSOLE_DIR ${MEGAMOL_DIR}/console)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# GLOBAL DLL DEFINITIONS
if(WIN32)
  add_definitions(-DGL_LOAD_DLL)
  add_definitions(-DMEGAMOLCORE_EXPORTS)
endif()

# VERSIONS
include(global_versioninfo)

# MEGAMOL OPTIONS
include(megamol_options)


###############################
# vislib
###############################
add_subdirectory(${MEGAMOL_VISLIB_DIR})
###############################
# core
###############################
add_subdirectory(${MEGAMOL_CORE_DIR})


###############################
# console
###############################
add_subdirectory(${MEGAMOL_CONSOLE_DIR})


###############################
# Plugins
###############################
#add_subdirectory(plugins)
