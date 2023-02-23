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
  if (MEGAMOL_SKIP_SHADER_INSTALL)
    set(CONFIG_SHADER_PATHS "mmAddShaderDir(\"${CMAKE_SOURCE_DIR}/core_gl/shaders\")") # TODO how to handle core_gl shaders?
    foreach (shader_path ${plugins_src_shader_paths})
      list(APPEND CONFIG_SHADER_PATHS "mmAddShaderDir(\"${shader_path}\")")
    endforeach ()
    list(JOIN CONFIG_SHADER_PATHS "\n" CONFIG_SHADER_PATHS)
  else ()
    set(CONFIG_SHADER_PATHS "mmAddShaderDir(basePath .. \"share/shaders\")")
  endif ()
else ()
  set(CONFIG_SHADER_PATHS "--mmAddShaderDir(basePath .. \"share/shaders\")")
endif ()
if (MEGAMOL_SKIP_RESOURCE_INSTALL)
  # TODO do resources already support file lookup in multiple directories like the shader factory?
  message(FATAL_ERROR "MEGAMOL_SKIP_RESOURCE_INSTALL not implemented!")
endif ()
mark_as_advanced(FORCE MEGAMOL_SKIP_RESOURCE_INSTALL) # TODO remove if error above is fixed.
configure_file(extra/megamol_config.lua.in ${CMAKE_BINARY_DIR}/megamol_config.lua @ONLY)
configure_file(extra/install_megamol_config_lua.cmake.in ${CMAKE_BINARY_DIR}/install_megamol_config_lua.cmake @ONLY)

install(SCRIPT ${CMAKE_BINARY_DIR}/install_megamol_config_lua.cmake)
