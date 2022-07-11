#
# MegaMol™ Main3000 Front End
# Copyright 2019, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_FRONTEND "build (new) frontend" ON)

if(BUILD_FRONTEND)
  project(frontend)
  set(BINARY_NAME megamol)

  require_external(libcxxopts)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  # Collect source files
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  set (resource_files "")
  if (MSVC)
    set(resource_files "${CMAKE_CURRENT_SOURCE_DIR}/extra/metadata.rc")
  endif()

  # Add target
  add_executable(${BINARY_NAME} ${header_files} ${source_files} "${resource_files}")
  target_include_directories(${BINARY_NAME} PRIVATE "src" "include")
  target_link_libraries(${BINARY_NAME} PRIVATE core frontend_services libcxxopts)
  target_link_libraries(${BINARY_NAME} PRIVATE ${CMAKE_DL_LIBS})

  if(MSVC)
    set_property(TARGET ${BINARY_NAME} APPEND_STRING PROPERTY LINK_FLAGS /STACK:8388608)
  endif()
  set_property(TARGET ${BINARY_NAME} APPEND_STRING PROPERTY LINK_FLAGS ${EXTERNAL_EXE_LINKER_FLAGS})

  # Grouping in Visual Studio
  set_target_properties(${BINARY_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

   # Installation rules for generated files
  install(TARGETS ${BINARY_NAME} RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
  if (WIN32)
    if(MSVC)
      install(FILES $<TARGET_PDB_FILE:${BINARY_NAME}> DESTINATION bin OPTIONAL)
    endif()
  endif ()
endif()

if(ENABLE_GL)
    set(CONFIG_GL_COMMENT "")
else()
    set(CONFIG_GL_COMMENT "--")
endif()
# Copy default configuration file
configure_file(extra/megamol_config.lua.in ${CMAKE_BINARY_DIR}/megamol_config.lua)
configure_file(extra/install_megamol_config_lua.cmake.in ${CMAKE_BINARY_DIR}/install_megamol_config_lua.cmake @ONLY)

install(SCRIPT ${CMAKE_BINARY_DIR}/install_megamol_config_lua.cmake)
