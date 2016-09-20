#
# MegaMol™ Console Front End
# Copyright 2015, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8.12)

project(MegaMolCon)
set(Building_MegaMolCon TRUE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
	message(STATUS "You should always specify a build type! Defaulting build type to 'Release'")
	set(CMAKE_BUILD_TYPE Release)
endif()


# finding main dependencies
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

find_package(glfw3 3.2 REQUIRED)
list(APPEND LIBS glfw)

find_package(AntTweakBar)
if (AntTweakBar_FOUND)
	message(STATUS "AntTweakBar will be used")
	add_definitions("-DHAS_ANTTWEAKBAR")
	list(APPEND LIBS ${AntTweakBar_LIBRARIES})
	include_directories(${AntTweakBar_INCLUDE_DIR})
endif()

# processor word size detection
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
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
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -D_DEBUG -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -D_NDEBUG -O3 -g0")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic") 


# collect version information
add_custom_target(MegaMolCon_versioninfo ALL
	perl VersionInfo.pl .
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	VERBATIM
	)

# Build MegaMol Core
# public header files for installation
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
set(header_files ${header_files} "src/version.gen.h")
# source files for compilation
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
#list(REMOVE_ITEM source_files
#	"src/moldyn/D3D11SimpleSphereRenderer.cpp"
#	)

include_directories(BEFORE "src")

add_custom_command(OUTPUT "src/version.gen.h"
	COMMAND
	DEPENDS "src/version.template.h" VersionInfo.pl)


# main target
add_executable(MegaMolCon ${header_files} ${source_files})
# Set target naming conventions for different build types
set_target_properties(MegaMolCon PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(MegaMolCon PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# Dependent projects only need to link against the core.so itself.
target_link_libraries(MegaMolCon PRIVATE ${LIBS})
add_dependencies(MegaMolCon MegaMolCon_versioninfo)

# install target
install(TARGETS MegaMolCon DESTINATION "bin/megamol")
if (AntTweakBar_FOUND)
	install(FILES ${AntTweakBar_LIBRARIES} DESTINATION "lib/megamol")
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/linkAntTweakBar.cmake.in ${CMAKE_BINARY_DIR}/linkAntTweakBar.cmake @ONLY)
	install(SCRIPT ${CMAKE_BINARY_DIR}/linkAntTweakBar.cmake)
endif()


# install target
set(cfg_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib/megamol")

set(cfg_MEGAMOLCON "${CMAKE_INSTALL_PREFIX}/bin/megamol/MegaMolCon")
set(MEGAMOL_SHELL_START_SCRIPT "megamol.sh")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	set(cfg_MEGAMOLCON "${CMAKE_INSTALL_PREFIX}/bin/megamol/MegaMolCond")
	set(MEGAMOL_SHELL_START_SCRIPT "megamold.sh")
endif()
configure_file(extra/megamol.sh ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} @ONLY)
install(PROGRAMS ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} DESTINATION "bin")

configure_file(extra/megamol.cfg ${CMAKE_BINARY_DIR}/megamol.cfg @ONLY)
configure_file(extra/install_megamol_cfg.cmake.in ${CMAKE_BINARY_DIR}/install_megamol_cfg.cmake @ONLY)
install(SCRIPT ${CMAKE_BINARY_DIR}/install_megamol_cfg.cmake)

