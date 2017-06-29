# Locate the glfw3 library
#
# This module defines the following variables:
#
# GLFW3_LIBRARIES    the name of the library;
# GLFW3_INCLUDE_DIRS where to find glfw include files.
# GLFW3_FOUND        true if both the GLFW3_LIBRARY and GLFW3_INCLUDE_DIR have been found.
#
# To help locate the library and include file, you can define a 
# variable called GLFW3_ROOT which points to the root of the glfw library 
# installation.
#

# default search dirs
set(_glfw3_HEADER_SEARCH_DIRS 
  "/usr/include"
  "/usr/local/include"
  "/opt/local/include"
  "$ENV{GLFW_ROOT}/include"
  "$ENV{GLFW3_ROOT}/include"
  "$ENV{PROGRAMFILES}/GLFW/include"
  "${GLFW_ROOT_DIR}/include"
  )
set(_glfw3_LIB_SEARCH_DIRS
  "/usr/lib"
  "/usr/local/lib"
  "/usr/lib64"
  "/usr/local/lib64"
  "/opt/local/lib"
  "$ENV{GLFW_ROOT}/lib"
  "$ENV{GLFW3_ROOT}/lib"
  "$ENV{PROGRAMFILES}/GLFW/lib"
  "${GLFW_ROOT_DIR}/lib"
  )

if(WIN32)
  find_path(GLFW3_INCLUDE_DIRS GLFW/glfw3.h
    ${_glfw3_HEADER_SEARCH_DIRS}
    DOC "The directory where GLFW/glfw3.h resides"
    )

  find_library(GLFW3_LIBRARY
    NAMES glfw3 GLFW
    PATHS
    ${_glfw3_LIB_SEARCH_DIRS}
    DOC "The GLFW library"
    )
else(WIN32)
  find_path(GLFW3_INCLUDE_DIRS GLFW/glfw3.h
    ${_glfw3_HEADER_SEARCH_DIRS}
    DOC "The directory where GLFW/glfw3.h resides"
    )
  # Prefer the static library.
  find_library(GLFW3_LIBRARY
    NAMES libGLFW.a GLFW libGLFW3.a GLFW3 libglfw.so libglfw.so.3 libglfw.so.3.0
    PATHS
    ${_glfw3_LIB_SEARCH_DIRS}
    DOC "The GLFW library"
    )
endif(WIN32)

set(GLFW3_FOUND "NO")
if(GLFW3_INCLUDE_DIRS AND GLFW3_LIBRARY)
  set(GLFW3_LIBRARIES ${GLFW3_LIBRARY})
  set(GLFW3_FOUND "YES")
  message(STATUS "Found GLFW")
endif(GLFW3_INCLUDE_DIRS AND GLFW3_LIBRARY)
