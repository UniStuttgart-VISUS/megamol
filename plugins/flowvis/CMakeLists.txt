#
# MegaMol™ flowvis Plugin
# Copyright 2021, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER flowvis EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build flowvis" OFF)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(flowvis)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_MESH_PLUGIN" CACHE INTERNAL "")

  # Find CGAL
  find_package(CGAL REQUIRED)

  if(NOT TARGET CGAL::CGAL)
    message(FATAL_ERROR "Target for CGAL not found")
  endif()

  if (TARGET CGAL)
    set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    set_target_properties(CGAL PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
  endif()

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "shaders/*")

  # Target definition
  add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files})
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                             "src")
  target_link_libraries(${PROJECT_NAME}
    PRIVATE core CGAL::CGAL
    PUBLIC mesh)

  if(MSVC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
  endif()

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ DESTINATION "share/shaders")

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Shaders" FILES ${shader_files})

  # Register plugin
  megamol_register_plugin(${PROJECT_NAME})
endif(BUILD_${EXPORT_NAME}_PLUGIN)
