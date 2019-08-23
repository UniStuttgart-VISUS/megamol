#
# Vislib glutTest
#
cmake_minimum_required(VERSION 3.0)

if(NOT DEFINED BUILDING_VISLIB_PROJECT)
  message(FATAL_ERROR "This CMakefile cannot be processed independently.")
endif()

#input file
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")

# target definition
add_executable(glutTest ${header_files} ${source_files})
target_include_directories(glutTest PUBLIC ${GLUT_INCLUDE_DIR})
target_link_libraries(glutTest vislib ${GLUT_LIBRARIES})
