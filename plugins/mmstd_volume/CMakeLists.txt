#
# MegaMol™ mmstd_volume Plugin
# Copyright 2015, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_MMSTD_VOLUME_PLUGIN "Option to build mmstd_VOLUME" ON)

if(BUILD_MMSTD_VOLUME_PLUGIN)
  project(mmstd_volume)

  add_subdirectory(datraw)
  
  set(MMSTD_VOLUME_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE INTERNAL "${PROJECT_NAME} include directories")

  set(LIBS vislib core glload datraw)
  include_directories(${VISLIB_INCLUDE_DIRS} ${CORE_INCLUDE_DIRS} ${GLLOAD_INCLUDE_DIRS} ${DATRAW_INCLUDE_DIRS})


  # Build MegaMol Plugin
  # public header files for installation
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  # source files for compilation
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  if(UNIX)
    list(REMOVE_ITEM source_files
      "src/dllmain.cpp"
      )
  endif()
  # shader files for installation
  file(GLOB_RECURSE shaders_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  include_directories("include" "src" "datraw")

  source_group("Header Files" FILES "include/*.h src/*.h")
  source_group("Source Files" FILES "src/*.cpp")
  source_group("Shaders" FILES "Shaders/*")
  
  # Target definition
  add_library(${PROJECT_NAME} SHARED ${header_files} ${source_files})
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

  string(TOUPPER ${PROJECT_NAME} EXPORT_NAME)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")

  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  endif()

endif(BUILD_MMSTD_VOLUME_PLUGIN)
