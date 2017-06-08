#
# MegaMol™ mmstd_moldyn Plugin
# Copyright 2015, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8.12)

project(mmstd_moldyn)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
	message(STATUS "You should always specify a build type! Defaulting build type to 'Release'")
	set(CMAKE_BUILD_TYPE Release)
endif()


# search for 3rd party libs
set(MegaMolCore_search_hints)
if (MegaMolCore_DIR)
	string(TOLOWER "${CMAKE_BUILD_TYPE}" lower_build_type)
	list(APPEND MegaMolCore_search_hints "${MegaMolCore_DIR}" "${MegaMolCore_DIR}/build.${lower_build_type}" "${MegaMolCore_DIR}/share/cmake/MegaMolCore")
	message(STATUS "Looking for MegaMolCore with hints: ${MegaMolCore_search_hints}")
endif()
find_package(MegaMolCore REQUIRED HINTS ${MegaMolCore_search_hints})
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
string(TOUPPER ${CMAKE_PROJECT_NAME} PROJECT_NAME_UPPER)
#message(STATUS "${CMAKE_PROJECT_NAME} ${PROJECT_NAME_UPPER}")
add_definitions(-Wall -pedantic -ansi -fPIC -DUNIX -D_GNU_SOURCE -D_LIN${BITS} -D${PROJECT_NAME_UPPER}_EXPORTS)
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
	message(STATUS "Version < 4.7")
	add_definitions(-std=c++0x)
else()
	add_definitions(-std=c++11)
endif()

# Set CXX flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -D_DEBUG -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -D_NDEBUG -O3 -g0")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic") 

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
add_library(${CMAKE_PROJECT_NAME} SHARED ${header_files} ${source_files})
# Set target naming conventions for different build types
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PREFIX "")
# Dependent projects only need to link against the core.so itself.
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${LIBS})


# Installation rules for generated files
set(package_name "${CMAKE_PROJECT_NAME}")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION "lib/megamol" EXPORT ${CMAKE_PROJECT_NAME}-target)
# Export the target to be used in the configuration file for find_package
install(EXPORT ${CMAKE_PROJECT_NAME}-target DESTINATION share/cmake/${package_name})
# Install configure script
set(cfg_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if ("${cfg_CMAKE_BUILD_TYPE}" STREQUAL "")
	set(cfg_CMAKE_BUILD_TYPE Release)
endif()
set(cfg_MegaMolCore_DIR ${MegaMolCore_DIR})
configure_file(cmake/ConfigInstall.cmake.in
	${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigInstall.cmake
	@ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigInstall.cmake DESTINATION share/cmake/${package_name} RENAME ${CMAKE_PROJECT_NAME}Config.cmake)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/megamol/${package_name}/Shaders")
set(package_install_dir "${CMAKE_INSTALL_PREFIX}/share/cmake/${package_name}")
# Install search directory into the cmake package user registry
include(${vislib_DIR}/cmake_package_register_utility.cmake)
vislibutil_prepare_cmake_package_user_repository("${package_name}" "${package_install_dir}" package_registry_file package_user_registry)
message(STATUS "Install will register ${package_registry_file} at ${package_user_registry}")
install(FILES ${package_registry_file} DESTINATION ${package_user_registry})
# TODO: Add or update megamol.cfg file (?)

if (${register_build_trees})
	# Export the build tree of this package. This makes it possible to include
	# this package without installing. Also place a version config file for cmake
	# to decide whether to take release or debug accordingly.
	# Set our known paths
	set(cfg_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
	set(cfg_SHADERS_DIRS ${CMAKE_SOURCE_DIR}/Shaders)
	set(cfg_MegaMolCore_DIR ${MegaMolCore_DIR})
	set(cfg_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
	if ("${cfg_CMAKE_BUILD_TYPE}" STREQUAL "")
		set(cfg_CMAKE_BUILD_TYPE Release)
	endif()
	# Configure our local config script
	configure_file(cmake/ConfigLocal.cmake.in
		${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake
		@ONLY)
	# Configure our local version script
	configure_file(cmake/ConfigVersion.cmake.in
		${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake
		@ONLY)
	export(TARGETS ${CMAKE_PROJECT_NAME} FILE ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}-target.cmake)
	export(PACKAGE ${CMAKE_PROJECT_NAME})
endif()

