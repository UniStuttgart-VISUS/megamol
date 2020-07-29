#
# MegaMol™ Render API modules
# Copyright 2019, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_RAPI "build render APIs" ON)

if(BUILD_RAPI)
  project(render_api)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  
  require_external(glad)


  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  #set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h" "include/*.hpp" "opengl_glfw/*.h"  "opengl_glfw/*.hpp" "service_collection/*.hpp")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "opengl_glfw/*.cpp" "service_collection/*.cpp")

  # Add target
  add_library(${PROJECT_NAME} STATIC ${header_files} ${source_files})
  set_target_properties(${PROJECT_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME ${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME} PUBLIC "opengl_glfw" "include" "service_collection")

  add_library(abstract_render_api INTERFACE)
  target_include_directories(abstract_render_api INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
  )
  target_link_libraries(abstract_render_api INTERFACE module_resources)
  
  #if(WIN32)
  #  set(VERSION_LIBRARY "Version.lib")
  #endif()

  # GLFW
  if(USE_GLFW)
    require_external(glfw3)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw3)
  endif()

  target_link_libraries(${PROJECT_NAME} PRIVATE ${VERSION_LIBRARY} ${CMAKE_DL_LIBS} PUBLIC abstract_render_api vislib glad)

  ## Configuration of the megamol.cfg megamol.sh remoteconsole.sh files
  #set(cfg_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib")
  #set(cfg_MEGAMOLCON "${CMAKE_INSTALL_PREFIX}/bin/${PROJECT_NAME}")

  #if(UNIX)
  #  include(GNUInstallDirs)
  #  set(cfg_EXTERNAL_LIB_PATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  #  set(MEGAMOL_SHELL_START_SCRIPT "megamol.sh")
  #  configure_file(extra/megamol.sh.in ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} @ONLY)
  #  install(PROGRAMS ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} DESTINATION "bin")
  #endif()

  #set(MMPLG_DIR "lib")
  #if(WIN32)
  #  set(MMPLG_DIR "bin")
  #endif()
  
  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

endif()
