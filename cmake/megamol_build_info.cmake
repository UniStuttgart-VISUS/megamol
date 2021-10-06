# megamol_build_info
# This script will provide several build information as C++ library.
# Usage: link to the megamol_build_info library as normal library and `#include "megamol_build_info.h"`.

set(INFO_SRC_DIR "${CMAKE_SOURCE_DIR}/cmake/megamol_build_info")
set(INFO_BIN_DIR "${CMAKE_BINARY_DIR}/megamol_build_info")

add_custom_target(megamol_build_info_script
  COMMAND ${CMAKE_COMMAND} -P ${INFO_SRC_DIR}/megamol_build_info_script.cmake)

configure_file(${INFO_SRC_DIR}/megamol_build_info.h ${INFO_BIN_DIR}/megamol_build_info.h COPYONLY)

# Dummy copy of the buildtime sources to not trigger missing file in library definition
configure_file(${INFO_SRC_DIR}/megamol_build_info_buildtime.cpp.in ${INFO_BIN_DIR}/megamol_build_info_buildtime.cpp COPYONLY)

# Configure the configure time constants
configure_file(${INFO_SRC_DIR}/megamol_build_info_configuretime.cpp.in ${INFO_BIN_DIR}/megamol_build_info_configuretime.cpp @ONLY)

# Define the library
add_library(megamol_build_info OBJECT
  ${INFO_BIN_DIR}/megamol_build_info.h
  ${INFO_BIN_DIR}/megamol_build_info_buildtime.cpp
  ${INFO_BIN_DIR}/megamol_build_info_configuretime.cpp)
target_compile_features(megamol_build_info PUBLIC cxx_std_17)
set_target_properties(megamol_build_info PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(megamol_build_info INTERFACE ${INFO_BIN_DIR})

add_dependencies(megamol_build_info megamol_build_info_script)
