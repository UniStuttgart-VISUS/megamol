#
# MegaMol™ mmstd.moldyn Plugin
# Copyright 2015, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8)

project(mmstd_moldyn)

# search for 3rd party libs
find_package(MegaMolCore REQUIRED HINTS ${MegaMolCore_DIR})
message(STATUS "MegaMolCore suggests vislib at: ${MegaMolCore_vislib_DIR}")
message(STATUS "MegaMolCore suggests install prefix: ${MegaMolCore_INSTALL_PREFIX}")
if (USE_MEGAMOLCORE_INSTALL_PREFIX)
	set(CMAKE_INSTALL_PREFIX ${MegaMolCore_INSTALL_PREFIX})
	message(STATUS "Using MegaMolCore install prefix")
endif()
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
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
# source files for compilation
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
list(REMOVE_ITEM source_files
	"src/dllmain.cpp"
	)
# shader files for installation
file(GLOB_RECURSE shaders_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

include_directories("include" "src")

# Target definition
add_library(mmstd_moldyn SHARED ${header_files} ${source_files})
# Set target naming conventions for different build types
set_target_properties(mmstd_moldyn PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(mmstd_moldyn PROPERTIES SUFFIX ".mmplg")
# Dependent projects only need to link against the core.so itself.
target_link_libraries(mmstd_moldyn PRIVATE ${LIBS})


# Installation rules for generated files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
install(TARGETS mmstd_moldyn DESTINATION "lib/megamol" EXPORT mmstd_moldyn-target)
# Export the target to be used in the configuration file for find_package
install(EXPORT mmstd_moldyn-target DESTINATION share/mmstd_moldyn)
# Install configure script
set(cfg_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if ("${cfg_CMAKE_BUILD_TYPE}" STREQUAL "")
	set(cfg_CMAKE_BUILD_TYPE Release)
endif()
set(cfg_mmstd_moldyn_MegaMolCore_DIR ${MegaMolCore_DIR})
configure_file(mmstd_moldynConfigInstall.cmake.in
	${CMAKE_BINARY_DIR}/mmstd_moldynConfigInstall.cmake
	@ONLY)
install(FILES ${CMAKE_BINARY_DIR}/mmstd_moldynConfigInstall.cmake DESTINATION share/mmstd_moldyn RENAME mmstd_moldynConfig.cmake)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/megamol/mmstd_moldyn/Shaders")
# TODO: Add entry to the package registry
# TODO: Add or update megamol.cfg file (?)

if (${register_build_trees})
	# Export the build tree of this package. This makes it possible to include
	# this package without installing. Also place a version config file for cmake
	# to decide whether to take release or debug accordingly.
	# Set our known paths
	set(cfg_mmstd_moldyn_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
	set(cfg_mmstd_moldyn_SHADERS_DIRS ${CMAKE_SOURCE_DIR}/Shaders)
	set(cfg_mmstd_moldyn_MegaMolCore_DIR ${MegaMolCore_DIR})
	set(cfg_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
	if ("${cfg_CMAKE_BUILD_TYPE}" STREQUAL "")
		set(cfg_CMAKE_BUILD_TYPE Release)
	endif()
	# Configure our local config script
	configure_file(mmstd_moldynConfigLocal.cmake.in
		${CMAKE_BINARY_DIR}/mmstd_moldynConfig.cmake
		@ONLY)
	# Configure our local version script
	configure_file(mmstd_moldynConfigVersion.cmake.in
		${CMAKE_BINARY_DIR}/mmstd_moldynConfigVersion.cmake
		@ONLY)
	export(TARGETS mmstd_moldyn FILE ${CMAKE_BINARY_DIR}/mmstd_moldyn-target.cmake)
	export(PACKAGE mmstd_moldyn)
endif()

