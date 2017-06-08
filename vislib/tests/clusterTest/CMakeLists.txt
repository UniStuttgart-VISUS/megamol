#
# Vislib clusterTest
#
cmake_minimum_required(VERSION 2.8)

if (NOT DEFINED BUILDING_VISLIB_PROJECT)
	message(FATAL_ERROR "This CMakefile cannot be processed independently.")
endif()

#input file
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")

# target definition
add_executable(clusterTest ${header_files} ${source_files})
target_link_libraries(clusterTest vislib ${GLUT_LIBRARIES} ${GLLOAD_LIBRARIES})
