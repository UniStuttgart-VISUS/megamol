#
# MegaMol™ Main3000 Front End
# Copyright 2019, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_MAIN3000 "build main3000" ON)

if(BUILD_MAIN3000)
  project(${MAIN3000_NAME})

  require_external(libcxxopts)

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
  target_link_libraries(${PROJECT_NAME} PRIVATE core frontend_services libcxxopts)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

   # Installation rules for generated files
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
  if (WIN32)
    if(MSVC)
      install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
    endif()
  endif ()
  
  if(UNIX)
    include(GNUInstallDirs)
    set(cfg_LIB_PATH "../lib") # relative path from bin to lib dir
    set(MAIN3000_SHELL_START_SCRIPT "mmmain3000.sh")
    configure_file(extra/mmmain3000.sh.in ${CMAKE_BINARY_DIR}/${MAIN3000_SHELL_START_SCRIPT} @ONLY)
    install(PROGRAMS ${CMAKE_BINARY_DIR}/${MAIN3000_SHELL_START_SCRIPT} DESTINATION "bin")
  endif()  

endif()
