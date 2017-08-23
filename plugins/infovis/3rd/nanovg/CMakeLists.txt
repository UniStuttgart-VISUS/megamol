#
# MegaMol™ nanovg dependency
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

project(nanovg)

# public header files for installation
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
# source files for compilation
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c")

include_directories("include" "src")

# Target definition
add_library(${PROJECT_NAME} STATIC ${header_files} ${source_files})
# Dependent projects only need to link against the core.so itself.
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER external)
