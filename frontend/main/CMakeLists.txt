# MegaMol
# Copyright (c) 2019, MegaMol Dev Team
# All rights reserved.
#

# Depedencies
find_package(cxxopts)

# Collect source files
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
set(resource_files "")
if (MSVC)
  set(resource_files "${CMAKE_CURRENT_SOURCE_DIR}/extra/metadata.rc")
endif ()

# Add target
add_executable(megamol ${header_files} ${source_files} "${resource_files}")
target_include_directories(megamol PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(megamol
  PRIVATE
    core
    frontend_services
    cxxopts::cxxopts
    ${CMAKE_DL_LIBS})

if (MSVC)
  target_link_options(megamol PUBLIC "/STACK:8388608")
endif ()

# Grouping in Visual Studio
set_target_properties(megamol PROPERTIES FOLDER base)
source_group("Header Files" FILES ${header_files})
source_group("Source Files" FILES ${source_files})

# Installation rules for generated files
install(TARGETS megamol
  RUNTIME DESTINATION "bin"
  ARCHIVE DESTINATION "lib")

if (WIN32 AND MSVC)
  install(FILES "$<TARGET_PDB_FILE:megamol>" DESTINATION bin OPTIONAL)
endif ()

# Copy default configuration file
if (MEGAMOL_USE_OPENGL)
  set(CONFIG_GL_COMMENT "")
else ()
  set(CONFIG_GL_COMMENT "--")
endif ()
configure_file(extra/megamol_config.lua.in ${CMAKE_BINARY_DIR}/megamol_config.lua)
configure_file(extra/install_megamol_config_lua.cmake.in ${CMAKE_BINARY_DIR}/install_megamol_config_lua.cmake @ONLY)

install(SCRIPT ${CMAKE_BINARY_DIR}/install_megamol_config_lua.cmake)
