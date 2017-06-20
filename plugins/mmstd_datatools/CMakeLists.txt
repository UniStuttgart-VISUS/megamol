#
# MegaMol™ mmstd_datatools Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_MMSTD_DATATOOLS_PLUGIN "Option to build mmstd_datatools" ON)

if(BUILD_MMSTD_DATATOOLS_PLUGIN)
  project(mmstd_datatools)

  set(MMSTD_DATATOOLS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE "" INTERNAL)

  find_package(OpenMP)

  set(LIBS vislib core glload)
  include_directories(${VISLIB_INCLUDE_DIRS} ${CORE_INCLUDE_DIRS} ${GLLOAD_INCLUDE_DIRS})
  if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()

  
  # Build MegaMol Plugin
  # public header files for installation
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  # source files for compilation
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  # shader files for installation
  file(GLOB_RECURSE shaders_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  include_directories("include" "src" "3rd/nanoflann")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${header_files} ${source_files})
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

  string(TOUPPER ${PROJECT_NAME} EXPORT_NAME)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")

  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  endif()


endif(BUILD_MMSTD_DATATOOLS_PLUGIN)
