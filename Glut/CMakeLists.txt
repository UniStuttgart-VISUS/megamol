#
# MegaMol™ Console Front End - Glut
# Copyright 2015, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8)

if (NOT DEFINED Building_MegaMolCon)
	message(FATAL_ERROR "This CMakefile cannot be processed independently.")
endif()

find_package(visglut REQUIRED)

set(LIBS ${vislib_LIBRARIES} ${MegaMolCore_LIBRARIES} ${visglut_LIBRARIES})
include_directories(${vislib_INCLUDE_DIRS} ${MegaMolCore_INCLUDE_DIRS} ${visglut_INCLUDE_DIR})

# collect version information
add_custom_target(MegaMolCon_GLUT_versioninfo ALL
	perl VersionInfo.pl .
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	VERBATIM
	)


# public header files for installation
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
# source files for compilation
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
list(REMOVE_ITEM source_files
	"NVSwapGroupController.cpp"
	)

# additional set up
add_definitions(-DMEGAMOLVIEWER_EXPORTS -DMEGAMOLVIEWER_USESTATIC)
#message(STATUS "Viewer_DIR == ${MegaMolCon_Viewer_DIR}")
include_directories(${MegaMolCon_Viewer_DIR}/include)
add_definitions(${MegaMolCon_common_definitions})
# Set CXX flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -D_DEBUG -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -D_NDEBUG -O3 -g0")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic") 

# Target definition
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_library(MegaMolGlut SHARED ${header_files} ${source_files})
# Set target naming conventions for different build types
set_target_properties(MegaMolGlut PROPERTIES DEBUG_POSTFIX "d")
# Dependent projects only need to link against the core.so itself.
target_link_libraries(MegaMolGlut PRIVATE ${LIBS})
add_dependencies(MegaMolGlut MegaMolCon_GLUT_versioninfo)

