#
# MegaMol™ infovis Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER infovis EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build infovis" ON)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(infovis)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_MMSTD_DATATOOLS_PLUGIN" CACHE INTERNAL "")

  # Add "Eigen" External
  include(ExternalProject)
  set(EIGEN_VER "3_3_4")
  string(REPLACE "_" "." EIGEN_TAG ${EIGEN_VER})
  set(EIGEN_INSTALL_CMD ${CMAKE_COMMAND})
  set(EIGEN_INSTALL_ARGS -E copy_directory ${PROJECT_BINARY_DIR}/extlibeigen-prefix/src/extlibeigen/Eigen ${EXTERNAL_INSTALL_LOCATION}/include/Eigen)
  
  ExternalProject_Add(extlibeigen
    GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
    GIT_TAG ${EIGEN_TAG}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${EIGEN_INSTALL_CMD} ${EIGEN_INSTALL_ARGS})

  add_library(Eigen INTERFACE)
  add_dependencies(Eigen extlibeigen)

  
  set(LIBS vislib core glad mmstd_datatools Eigen)
  include_directories(${EXTERNAL_INSTALL_LOCATION}/include "include" "src")

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_dependencies(${PROJECT_NAME} vislib glad core mmstd_datatools)
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Shaders" FILES ${shader_files})

  # Format
  add_clang_format(${PROJECT_NAME} 
    STYLE "${CMAKE_SOURCE_DIR}/.clang-format" 
    FILES ${public_header_files} ${header_files} ${source_files})

endif(BUILD_${EXPORT_NAME}_PLUGIN)