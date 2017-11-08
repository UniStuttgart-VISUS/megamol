#
# LIBTESS2 - Modified Version for MegaMol
#

cmake_minimum_required(VERSION 3.2)

option(BUILD_LIBTESS2 "Build LIBTESS2" ON)

if(BUILD_LIBTESS2)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}")
project(libtess2 C)

include_directories("include" "src")

file(GLOB libtess2_header_files "include/*.h")
file(GLOB libtess2_source_files "src/*.h" "src/*.c")

add_library(${PROJECT_NAME} ${libtess2_header_files} ${libtess2_source_files})
	
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
source_group("Header Files" FILES ${libtess2_header_files})
source_group("Source Files" FILES ${libtess2_source_files})

install(TARGETS ${PROJECT_NAME} 
  DESTINATION lib)

endif(BUILD_LIBTESS2)