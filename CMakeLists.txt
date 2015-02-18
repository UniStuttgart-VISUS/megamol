#
# MegaMol™ mmstd.moldyn Plugin
# Copyright 2015, by MegaMol Consortium
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8)

project(mmstd.moldyn)

# search for 3rd party libs
find_package(MegaMolCore REQUIRED)
find_package(vislib REQUIRED HINTS ${MegaMolCore_vislib_DIR})

set(LIBS ${vislib_LIBRARIES} ${MegaMolCore_LIBRARIES})
include_directories(${vislib_INCLUDE_DIRS} ${MegaMolCore_INCLUDE_DIRS})

# processor word size detection
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(BITS 64)
else()
	set(BITS 32)
endif()

# compiler options
add_definitions(-Wall -pedantic -ansi -fPIC -DUNIX -D_GNU_SOURCE -D_LIN${BITS} -DMMSTD_MOLDYN_EXPORTS)
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
	message(STATUS "Version < 4.7")
	add_definitions(-std=c++0x)
else()
	add_definitions(-std=c++11)
endif()

# Set CXX flags for debug mode
set(CMAKE_CXX_FLAGS_DEBUG -DDEBUG)

# Build MegaMol Plugin
# public header files for installation
#file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
set(header_files "mmstd.moldyn.h" "stdafx.h" "targetver.h")
set(source_files "mmstd.moldyn.cpp" "stdafx.cpp" "dllmain.cpp")
# source files for compilation
#file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
#list(REMOVE_ITEM source_files
#	"src/moldyn/D3D11SimpleSphereRenderer.cpp"
#	"src/special/AbstractStereoDisplay.cpp"
#	"src/special/ColStereoDisplay.cpp"
#	"src/utility/AbstractD3D11RenderObject.cpp"
#	"src/utility/D3D11BoundingBox.cpp"
#	"src/view/CallUpdateDirect3D.cpp"
#	"src/view/ViewDirect3D.cpp"
#	"src/vismol2/Mol20DataCall.cpp"
#	"src/vismol2/Mol20DataSource.cpp"
#	"src/vismol2/Mol20Renderer.cpp"
#	"src/vismol2/Mol2Data.cpp"
#	)
#include_directories("include" "src")

# Target definition
add_library(mmstd.moldyn SHARED ${header_files} ${source_files})
# Set target naming conventions for different build types
set_target_properties(mmstd.moldyn PROPERTIES DEBUG_POSTFIX "d")
# Dependent projects only need to link against the core.so itself.
target_link_libraries(mmstd.moldyn PRIVATE ${LIBS})


# Installation rules for generated files
# TODO: fix the install rules::
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
#install(TARGETS MegaMolCore DESTINATION "lib/mmcore" EXPORT MegaMolCore-target)
# Export the target to be used in the configuration file for find_package
#install(EXPORT MegaMolCore-target DESTINATION share/mmcore)
# Install configure script
#install(FILES MegaMolCoreConfig.cmake DESTINATION share/mmcore)
# TODO: Add entry to the package registry

# Export the build tree of this package. This makes it possible to include
# this package without installing. Also place a version config file for cmake
# to decide whether to take release or debug accordingly.
# Set our known paths
#set(cfg_MegaMolCore_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
#set(cfg_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
#if ("${cfg_CMAKE_BUILD_TYPE}" STREQUAL "")
#	set(cfg_CMAKE_BUILD_TYPE Release)
#endif()
# Configure our local config script
#configure_file(MegaMolCoreConfigLocal.cmake.in
#	${CMAKE_BINARY_DIR}/MegaMolCoreConfig.cmake
#	@ONLY)
# Configure our local version script
#configure_file(MegaMolCoreConfigVersion.cmake.in
#	${CMAKE_BINARY_DIR}/MegaMolCoreConfigVersion.cmake
#	@ONLY)
#export(TARGETS MegaMolCore FILE ${CMAKE_BINARY_DIR}/MegaMolCore-target.cmake)
#export(PACKAGE MegaMolCore)

