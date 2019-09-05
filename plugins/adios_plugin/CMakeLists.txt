#
# MegaMol™ adios_plugin Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER adios_plugin EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build adios_plugin" OFF)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(adios_plugin)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_MMSTD_DATATOOLS_PLUGIN" CACHE INTERNAL "")

  option(USE_EXTERNAL_ADIOS "Option to use external ADIOS2" OFF)
  if (USE_EXTERNAL_ADIOS)
    find_package(ADIOS2 REQUIRED)
  else()
    require_external(adios2)
  endif()


  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  #file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
    # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_link_libraries(${PROJECT_NAME} PRIVATE core mmstd_datatools)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src")

  if(USE_EXTERNAL_ADIOS)
    target_link_libraries(${PROJECT_NAME} PRIVATE adios2::adios2)
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE adios2)
  endif()

  if(MPI_C_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_C_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})
  endif()

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")

  # if(UNIX)
  #   include(GNUInstallDirs)
  #   get_target_property(adios_lib adios2::adios2 IMPORTED_LOCATION_RELEASE)
  #   get_filename_component(dir ${adios_lib} DIRECTORY)
  #   file(GLOB ADIOS_LIB_FILES "${dir}/lib*")
  #   install(FILES ${ADIOS_LIB_FILES} DESTINATION ${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR})
  # endif()

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  if(NOT USE_EXTERNAL_ADIOS)
    install_external(TARGETS adios2)
  endif()
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

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
    string(REGEX REPLACE "^include\\\\adios_plugin" "Public Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
endif(BUILD_${EXPORT_NAME}_PLUGIN)
