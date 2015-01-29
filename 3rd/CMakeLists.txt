#
# Vislib 3rd Party Libraries
#
cmake_minimum_required(VERSION 2.8)


# build glload
add_custom_target(glload ALL
	./build.sh
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glload
	)

set(GLLOAD_LIBRARIES_MINE ${CMAKE_CURRENT_SOURCE_DIR}/glload/build/libglload.so)
set(GLLOAD_INCLUDE_DIRS_MINE "${CMAKE_CURRENT_SOURCE_DIR}/glload/include")
 	
set(GLLOAD_LIBRARIES "${GLLOAD_LIBRARIES_MINE}" PARENT_SCOPE)
set(GLLOAD_INCLUDE_DIRS "${GLLOAD_INCLUDE_DIRS_MINE}" PARENT_SCOPE)

# Set installation rules
install(DIRECTORY "${GLLOAD_INCLUDE_DIRS_MINE}/" DESTINATION "include")
install(FILES "${GLLOAD_LIBRARIES_MINE}" DESTINATION "lib")

