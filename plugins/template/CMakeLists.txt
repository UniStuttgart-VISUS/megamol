#
# MegaMol™ MegaMolPlugin Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER MegaMolPlugin EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build MegaMolPlugin" ON)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(MegaMolPlugin)

  set(LIBS vislib core glload)
  include_directories(${VISLIB_INCLUDE_DIRS} ${CORE_INCLUDE_DIRS} ${GLLOAD_INCLUDE_DIRS})

  # Build MegaMol Plugin
  # source files for compilation
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

  include_directories(BEFORE "include" "src")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${header_files} ${source_files})
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")

  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
    source_group("Header Files" FILES "include/*.h src/*.h")
    source_group("Source Files" FILES "src/*.cpp")
    #source_group("Shaders" FILES "Shaders/*")
  endif()


endif(BUILD_${EXPORT_NAME}_PLUGIN)