#
# LIBTESS2 - Modified Version for MegaMol
#

project(libtess2)

# public header files for installation
file(GLOB_RECURSE libtess2_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
# source files for compilation
file(GLOB_RECURSE libtess2_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h" "src/*.c")

include_directories("include" "src")

# Target definition
add_library(${PROJECT_NAME} STATIC ${libtess2_header_files} ${libtess2_source_files})
# Dependent projects only need to link against the core.so itself.
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
	
source_group("Header Files" FILES ${libtess2_header_files})
source_group("Source Files" FILES ${libtess2_source_files})

install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER external)