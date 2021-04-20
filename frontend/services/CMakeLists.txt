#
# MegaMol™ Frontend Services
# Copyright 2019, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_FRONTEND_SERVICES "build frontend services" ON)

if(BUILD_FRONTEND_SERVICES)
  project(frontend_services)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  
  require_external(glad)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  #set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "include/*.h"
    "include/*.hpp"
    "cuda/*h"
    "cuda/*hpp"
    "opengl_glfw/*.h"
    "opengl_glfw/*.hpp"
    "service_collection/*.hpp"
    "gui/*.hpp"
    "lua_service_wrapper/*.hpp"
    "screenshot_service/*.hpp"
    "framestatistics_service/*.hpp"
    "project_loader/*.hpp"
    "image_presentation/*.hpp"
#   "service_template/*.hpp"
    )
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "cuda/*cpp"
    "opengl_glfw/*.cpp"
    "service_collection/*.cpp"
    "gui/*.cpp"
    "lua_service_wrapper/*.cpp"
    "screenshot_service/*.cpp"
    "framestatistics_service/*.cpp"
    "project_loader/*.cpp"
    "image_presentation/*.cpp"
#   "service_template/*.cpp"
    )

  # Add target
  add_library(${PROJECT_NAME} STATIC ${header_files} ${source_files})
  set_target_properties(${PROJECT_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME ${PROJECT_NAME})

  target_include_directories(${PROJECT_NAME} PUBLIC
    "include"
    "cuda"
    "opengl_glfw"
    "service_collection"
    "gui"
    "lua_service_wrapper"
    "screenshot_service"
    "framestatistics_service"
    "project_loader"
    "image_presentation"
#   "service_template"
    )

  add_library(abstract_frontend_service INTERFACE)
  target_include_directories(abstract_frontend_service INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
  )
  target_link_libraries(abstract_frontend_service INTERFACE frontend_resources)

  if(ENABLE_CUDA)
    find_package(CUDAToolkit)
    target_compile_definitions(${PROJECT_NAME} PUBLIC MM_CUDA_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver)
  endif()
  
  #if(WIN32)
  #  set(VERSION_LIBRARY "Version.lib")
  #endif()

  # GLFW
  if(USE_GLFW)
    require_external(glfw3)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw3)
  endif()

  target_link_libraries(${PROJECT_NAME} PRIVATE ${VERSION_LIBRARY} ${CMAKE_DL_LIBS} PUBLIC abstract_frontend_service vislib glad core gui)

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
