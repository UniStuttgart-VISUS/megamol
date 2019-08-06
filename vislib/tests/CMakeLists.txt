#
# Vislib test
#
cmake_minimum_required(VERSION 3.0)

if(NOT DEFINED BUILDING_VISLIB_PROJECT)
  message(FATAL_ERROR "This CMakefile cannot be processed independently.")
endif()

add_subdirectory(test)

find_package(GLUT)
if(GLUT_FOUND)
  add_subdirectory(glutTest)
  add_subdirectory(clusterTest)
endif(GLUT_FOUND)
