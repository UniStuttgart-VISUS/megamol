cmake_minimum_required(VERSION 3.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(megamol)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "default install path" FORCE)
endif()

# configure directories
set(MEGAMOL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MEGAMOL_VISLIB_DIR ${MEGAMOL_DIR}/vislib)
set(MEGAMOL_CORE_DIR ${MEGAMOL_DIR}/core)
set(MEGAMOL_CONSOLE_DIR ${MEGAMOL_DIR}/console)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# GLOBAL DLL DEFINITIONS
if(WIN32)
  add_definitions(-DGL_LOAD_DLL)
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
set(CONSOLE_NAME mmconsole)
add_subdirectory(${MEGAMOL_CONSOLE_DIR})

# add directory structure for visual studio
if(WIN32)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD TRUE)
endif()

###############################
# console
###############################
add_subdirectory(configurator)

###############################
# Plugins
###############################
add_subdirectory(plugins)


