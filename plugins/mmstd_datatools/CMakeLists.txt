#
# MegaMol™ mmstd_datatools Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_MMSTD_DATATOOLS_PLUGIN "Option to build mmstd_datatools" ON)

if(BUILD_MMSTD_DATATOOLS_PLUGIN)
  project(mmstd_datatools)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_GEOMETRY_CALLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")
  
  set(LIBS vislib core glad geometry_calls)

  include_directories("include" "src" "3rd/nanoflann")

  if (OPENMP_FOUND OR WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()

  include(ExternalProject)
  find_package(Git REQUIRED)

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tinyply-prefix/)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} clone https://github.com/ddiakopoulos/tinyply.git tinyply -b "2.1" --depth 1
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/tinyply-prefix/cmakeout.txt
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/tinyply-prefix/cmakeerr.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tinyply-prefix
  )

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")

  # Grouping in Visual Studio
  foreach(FILE_NAME ${source_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^src" "Source Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  foreach(FILE_NAME ${header_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^src" "Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  foreach(FILE_NAME ${public_header_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^include\\\\mmstd_datatools" "Public Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()

  source_group("tinyply" FILES "${CMAKE_CURRENT_BINARY_DIR}/tinyply-prefix/tinyply/source/tinyply.h" "${CMAKE_CURRENT_BINARY_DIR}/tinyply-prefix/tinyply/source/tinyply.cpp")

  list(APPEND source_files "${CMAKE_CURRENT_BINARY_DIR}/tinyply-prefix/tinyply/source/tinyply.cpp")
  list(APPEND header_files "${CMAKE_CURRENT_BINARY_DIR}/tinyply-prefix/tinyply/source/tinyply.h")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  add_dependencies(${PROJECT_NAME} vislib core)
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  string(TOUPPER ${PROJECT_NAME} EXPORT_NAME)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  
endif(BUILD_MMSTD_DATATOOLS_PLUGIN)
