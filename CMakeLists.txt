#
# VISlib
# Copyright (C) 2015 by Sebastian Grottel
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8)

project(vislib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

find_package(OpenGL REQUIRED)
set(CMAKE_THREAD_PREFER_PTHREAD)
find_package(Threads REQUIRED)
find_package(X11 REQUIRED)
find_package(Curses REQUIRED)
find_package(UUID REQUIRED)

# processor word size detection
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(BITS 64)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(BITS 32)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(OUT_SUBDIR "linux.debug")
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(OUT_SUBDIR "linux.release")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

# compiler options
if(GCC_VERSION VERSION_LESS 4.7)
	add_definitions(-std=c++0x -Wall -pedantic -ansi -fPIC -DUNIX -D_GNU_SOURCE -D_LIN${BITS} )
else()
	add_definitions(-std=c++11 -Wall -pedantic -ansi -fPIC -DUNIX -D_GNU_SOURCE -D_LIN${BITS} )
endif()

# build glload
add_custom_target(glload ALL
	./build.sh
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rd/glload
	)

# collect version information
add_custom_target(versioninfo ALL
	perl src/makevislibversion.pl . src/vislibversion.template.h include/vislib/vislibversion.h
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	VERBATIM
	)

# build vislib
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
include_directories("include" "src" "3rd/glload/include")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/${OUT_SUBDIR})

# target definition
add_library(vislib STATIC ${header_files} ${source_files})



# build tests
# TODO
