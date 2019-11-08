#
# MegaMol™ Remote Console
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

option(BUILD_REMOTECONSOLE "Build remote Lua console" ON)

if(BUILD_REMOTECONSOLE)
  project(remoteconsole)

  require_external(libzmq)
  require_external(libcppzmq)
  require_external(libcxxopts)

  # Glob remote console files.
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")

  # Setup target.
  add_executable(${PROJECT_NAME} ${header_files} ${source_files})
  target_include_directories(${PROJECT_NAME} 
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PUBLIC "include" "src")
  target_link_libraries(${PROJECT_NAME} PRIVATE lua libzmq libcppzmq libcxxopts)

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

  # Installation rules for generated files
  if(UNIX)
    include(GNUInstallDirs)
    set(cfg_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(cfg_EXTERNAL_LIB_PATH "${CMAKE_INSTALL_LIBDIR}")
    set(cfg_REMOTE "${CMAKE_INSTALL_PREFIX}/bin/${PROJECT_NAME}")
    set(MEGAMOL_REMOTECONSOLE_SCRIPT "remoteconsole.sh")
    configure_file(extra/remoteconsole.sh.in ${CMAKE_BINARY_DIR}/${MEGAMOL_REMOTECONSOLE_SCRIPT} @ONLY)
    install(PROGRAMS ${CMAKE_BINARY_DIR}/${MEGAMOL_REMOTECONSOLE_SCRIPT} DESTINATION "bin")
  endif()
  
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
endif(BUILD_REMOTECONSOLE)
