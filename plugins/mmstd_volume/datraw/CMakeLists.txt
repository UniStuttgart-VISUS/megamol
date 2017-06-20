#
# MegaMol™ datraw dependency
# Copyright 2015, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

project(datraw)

set(DATRAW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE "" INTERNAL)

# public header files for installation
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
# source files for compilation
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c")

include_directories("include" "src")

# Target definition
add_library(${PROJECT_NAME} STATIC ${header_files} ${source_files})
# Dependent projects only need to link against the core.so itself.
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")

if(WIN32)
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER external)
endif()

