#
# MegaMol™ Console Front End - Viewer
# Copyright 2015, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8)

if (NOT DEFINED Building_MegaMolCon)
	message(FATAL_ERROR "This CMakefile cannot be processed independently.")
endif()

# collect version information
add_custom_target(MegaMolCon_Viewer_versioninfo ALL
	perl VersionInfo.pl .
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	VERBATIM
	)

# generate dynamic import api
add_custom_command(
	OUTPUT include/MegaMolViewerDynamic.h
	COMMAND perl genDynIncl.pl include/MegaMolViewerStatic.h
	MAIN_DEPENDENCY include/MegaMolViewerStatic.h
	DEPENDS genDynIncl.pl MegaMolCon_Viewer_versioninfo
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	VERBATIM
	)

# main target
# Note: COMMAND is empty on purpose. This is just the main target to be specified from the main script
add_custom_target(MegaMolCon_Viewer
	COMMAND
	DEPENDS MegaMolCon_Viewer_versioninfo include/MegaMolViewerDynamic.h
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	VERBATIM
	)
set(MegaMolCon_Viewer_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

