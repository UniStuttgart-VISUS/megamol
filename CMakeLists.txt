#
# VISlib
# Copyright (C) 2015 by Sebastian Grottel
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8)

project(vislib)
set(BUILDING_VISLIB_PROJECT 1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#find_package(OpenGL REQUIRED)
set(CMAKE_THREAD_PREFER_PTHREAD)
find_package(Threads REQUIRED)
find_package(X11 REQUIRED)
find_package(Curses REQUIRED)
find_package(UUID REQUIRED)
find_package(OpenGL REQUIRED)
set(LIBS ${CMAKE_THREAD_LIBS_INIT} ${X11_LIBRARIES} ${CURSES_LIBRARIES} -lrt -lcrypt ${OPENGL_LIBRARIES})
include_directories(${OPENGL_INCLUDE_DIR})

# processor word size detection
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(BITS 64)
else()
	set(BITS 32)
endif()

# compiler options
add_definitions(-Wall -pedantic -ansi -fPIC -DUNIX -D_GNU_SOURCE -D_LIN${BITS})
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
    message(STATUS "Version < 4.7")
	add_definitions(-std=c++0x)
else()
	add_definitions(-std=c++11)
endif()

# Set CXX flags for debug mode
set(CMAKE_CXX_FLAGS_DEBUG -DDEBUG)

# Build 3rd party libraries
add_subdirectory(3rd)
include_directories(${GLLOAD_INCLUDE_DIRS})

# collect version information
add_custom_target(versioninfo ALL
	perl src/makevislibversion.pl . src/vislibversion.template.h include/vislib/vislibversion.h
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	VERBATIM
	)

# Build vislib
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
list(REMOVE_ITEM source_files
	"src/sys/DynamicFunctionPointer.cpp"
	"src/sys/PAMException.cpp"
	"src/sys/LinuxDaemon.cpp"
	"src/graphics/d3d/AbstractD3D11WindowImpl.cpp"
	"src/graphics/d3d/AbstractD3DAdapterInformation.cpp"
	"src/graphics/d3d/D3D9AdapterInformation.cpp"
	"src/graphics/d3d/D3D11Window.cpp"
	"src/graphics/d3d/D3DCamera.cpp"
	"src/graphics/d3d/D3DException.cpp"
	"src/graphics/d3d/D3DMatrix.cpp"
	"src/graphics/d3d/D3DPoint3D.cpp"
	"src/graphics/d3d/d3dutils.cpp"
	"src/graphics/d3d/D3DVector3D.cpp"
	"src/graphics/d3d/d3dverify.cpp"
	"src/graphics/d3d/D3DVISLogo.cpp"
	"src/graphics/gl/GlutMouseInteractionAdapter.cpp"
	)
include_directories("include" "src")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/linux)

# Target definition
add_library(vislib STATIC ${header_files} ${source_files})
# Set target naming conventions for different build types
set_target_properties(vislib PROPERTIES DEBUG_POSTFIX "d")

target_link_libraries(vislib ${LIBS})

# build tests if variable VISLIB_BUILD_TESTS is defined
if (DEFINED VISLIB_BUILD_TESTS)
	add_subdirectory(tests)
endif()


# Installation rules for generated files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
install(TARGETS vislib DESTINATION "lib/vislib" EXPORT vislib-target)
# Export the target to be used in the configuration file for find_package
install(EXPORT vislib-target DESTINATION share/vislib)
# Install configure script
install(FILES vislibConfig.cmake DESTINATION share/vislib)
# TODO: Add entry to the package registry

# Export the build tree of this package. This makes it possible to include
# this package without installing. Also place a version config file for cmake
# to decide whether to take release or debug accordingly.
# Set our known paths
set(cfg_glload_LIBRARIES ${GLLOAD_LIBRARIES})
set(cfg_vislib_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include ${GLLOAD_INCLUDE_DIRS})
set(cfg_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if ("${cfg_CMAKE_BUILD_TYPE}" STREQUAL "")
	set(cfg_CMAKE_BUILD_TYPE Release)
endif()
# Configure our local config script
configure_file(vislibConfigLocal.cmake.in
			${CMAKE_BINARY_DIR}/vislibConfig.cmake
			@ONLY)
# Configure our local version script
configure_file(vislibConfigVersion.cmake.in
			${CMAKE_BINARY_DIR}/vislibConfigVersion.cmake
			@ONLY)
export(TARGETS vislib FILE ${CMAKE_BINARY_DIR}/vislib-target.cmake)
export(PACKAGE vislib)

