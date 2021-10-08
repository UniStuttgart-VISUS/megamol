# megamol_build_info
# This script will provide several build information as C++ library.
# Usage: link to the megamol_build_info library as normal library and `#include "megamol_build_info.h"`.

# Used directories
set(INFO_SRC_DIR "${CMAKE_SOURCE_DIR}/cmake/megamol_build_info")
set(INFO_BIN_DIR "${CMAKE_BINARY_DIR}/megamol_build_info")

# Run build time script
add_custom_target(megamol_build_info_script
  COMMAND ${CMAKE_COMMAND} -P ${INFO_SRC_DIR}/megamol_build_info_script.cmake)

# Set configure time values
configure_file(${INFO_SRC_DIR}/megamol_build_info.h ${INFO_BIN_DIR}/megamol_build_info.h COPYONLY)
configure_file(${INFO_SRC_DIR}/megamol_build_info_configuretime.h.in ${INFO_BIN_DIR}/megamol_build_info_configuretime.h @ONLY)

# Define the library
add_library(megamol_build_info INTERFACE)
target_compile_features(megamol_build_info INTERFACE cxx_std_17)
target_include_directories(megamol_build_info INTERFACE ${INFO_BIN_DIR})

add_dependencies(megamol_build_info megamol_build_info_script)
