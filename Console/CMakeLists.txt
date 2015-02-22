#
# MegaMol™ Console Front End - Console
# Copyright 2015, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8)

if (NOT DEFINED Building_MegaMolCon)
	message(FATAL_ERROR "This CMakefile cannot be processed independently.")
endif()


set(LIBS ${vislib_LIBRARIES} ${MegaMolCore_LIBRARIES})
list(APPEND LIBS "dl")
include_directories(${vislib_INCLUDE_DIRS} ${MegaMolCore_INCLUDE_DIRS})


# collect version information
add_custom_target(MegaMolCon_CONSOLE_versioninfo ALL
	perl VersionInfo.pl .
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	VERBATIM
	)


# public header files for installation
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
# source files for compilation
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")

# additional set up
#message(STATUS "Viewer_DIR == ${MegaMolCon_Viewer_DIR}")
include_directories(${MegaMolCon_Viewer_DIR}/include)
add_definitions(${MegaMolCon_common_definitions})

# main target
add_executable(MegaMolCon ${header_files} ${source_files})
# Set target naming conventions for different build types
set_target_properties(MegaMolCon PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(MegaMolCon PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# Dependent projects only need to link against the core.so itself.
target_link_libraries(MegaMolCon PRIVATE ${LIBS})
add_dependencies(MegaMolCon MegaMolCon_CONSOLE_versioninfo)

