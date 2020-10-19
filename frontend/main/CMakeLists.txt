#
# MegaMol™ Main3000 Front End
# Copyright 2019, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_FRONTEND "build (new) frontend" ON)

if(BUILD_FRONTEND)
  project(frontend)
  set(BINANRY_NAME megamol)

  require_external(libcxxopts)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  # Collect source files
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

  # Add target
  add_executable(${BINANRY_NAME} ${header_files} ${source_files})
  target_include_directories(${BINANRY_NAME} PRIVATE "src" "include")
  target_link_libraries(${BINANRY_NAME} PRIVATE core frontend_services libcxxopts)
  target_link_libraries(${BINANRY_NAME} PRIVATE ${CMAKE_DL_LIBS})

  # Grouping in Visual Studio
  set_target_properties(${BINANRY_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

   # Installation rules for generated files
  install(TARGETS ${BINANRY_NAME} RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
  if (WIN32)
    if(MSVC)
      install(FILES $<TARGET_PDB_FILE:${BINANRY_NAME}> DESTINATION bin OPTIONAL)
    endif()
  endif ()
  
  if(UNIX)
    include(GNUInstallDirs)
    set(cfg_LIB_PATH "../lib") # relative path from bin to lib dir
    set(MEGAMOL_SHELL_START_SCRIPT "megamol.sh")
    configure_file(extra/megamol.sh.in ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} @ONLY)
    install(PROGRAMS ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} DESTINATION "bin")
  endif()  

endif()
