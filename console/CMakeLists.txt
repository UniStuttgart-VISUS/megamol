#
# MegaMol™ Console Front End
# Copyright 2017, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_CONSOLE "Build console" ON)

if(BUILD_CONSOLE)
  project(${CONSOLE_NAME})
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  # Collect source files
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

  # Add target
  add_executable(${PROJECT_NAME} ${header_files} ${source_files})
  set_target_properties(${PROJECT_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME ${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME} PRIVATE "src" "include")
  target_link_libraries(${PROJECT_NAME} PRIVATE core)

  if(WIN32)
    set(VERSION_LIBRARY "Version.lib")
  endif()

  # GLFW
  if(USE_GLFW)
    require_external(glfw3)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw3)
  endif()

  target_link_libraries(${PROJECT_NAME} PRIVATE ${VERSION_LIBRARY} ${CMAKE_DL_LIBS})

   # Installation rules for generated files
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
  if (WIN32)
    if(MSVC)
      install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
    endif()
  endif ()

  # Configuration of the megamol.cfg megamol.sh remoteconsole.sh files

  if(UNIX)
    include(GNUInstallDirs)
    set(cfg_LIB_PATH "../lib") # relative path from bin to lib dir
    set(MEGAMOL_SHELL_START_SCRIPT "megamol.sh")
    configure_file(extra/megamol.sh.in ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} @ONLY)
    install(PROGRAMS ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} DESTINATION "bin")
  endif()

  set(MMPLG_DIR "lib")
  if(WIN32)
    set(MMPLG_DIR "bin")
  endif()

  configure_file(extra/megamolconfig.lua.in ${CMAKE_BINARY_DIR}/megamolconfig.lua)
  configure_file(extra/install_megamolconfig_lua.cmake.in ${CMAKE_BINARY_DIR}/install_megamolconfig_lua.cmake @ONLY)

  install(SCRIPT ${CMAKE_BINARY_DIR}/install_megamolconfig_lua.cmake)
  
  # Grouping in Visual Studio
  set_target_properties(${CONSOLE_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

endif()
