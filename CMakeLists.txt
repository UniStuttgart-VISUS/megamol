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
add_subdirectory(${MEGAMOL_CONSOLE_DIR})

# add directory structure for visual studio
if(WIN32)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  if(BUILD_VISLIB)
    set_target_properties(glload PROPERTIES FOLDER external)
    set_target_properties(vislib PROPERTIES FOLDER base)
  endif()
  if(BUILD_CORE)
    set_target_properties(core PROPERTIES FOLDER base)
  endif()
  if(BUILD_CONSOLE)
    set_target_properties(console PROPERTIES FOLDER base)
  endif()
endif()

###############################
# Plugins
###############################
add_subdirectory(plugins)


