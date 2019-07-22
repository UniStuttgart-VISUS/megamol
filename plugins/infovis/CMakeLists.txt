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

  # Add Delaunator.
  add_external_project(Delau_ext
    GIT_REPOSITORY https://github.com/delfrrr/delaunator-cpp.git
    GIT_TAG "v0.4.0"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND "")
  add_external_library(Delau INTERFACE
    DEPENDS Delau_ext
    INCLUDE_DIR "src/Delau_ext/include")

  # Add Barnes-Hut t-SNE.
  add_external_project(bhtsne_ext
    GIT_REPOSITORY  https://github.com/lvdmaaten/bhtsne.git
    GIT_TAG "36b169c88250d0afe51828448dfdeeaa508f13bc" 
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/bhtsne/CMakeLists.txt"
      "<SOURCE_DIR>/CMakeLists.txt")
  add_external_library(bhtsne STATIC
    DEPENDS bhtsne_ext
    INCLUDE_DIR "src/bhtsne_ext"
    LIBRARY_DEBUG "lib/${CMAKE_STATIC_LIBRARY_PREFIX}bhtsned${CMAKE_STATIC_LIBRARY_SUFFIX}"
    LIBRARY_RELEASE "lib/${CMAKE_STATIC_LIBRARY_PREFIX}bhtsne${CMAKE_STATIC_LIBRARY_SUFFIX}")

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PUBLIC "include" "src")
  target_link_libraries(${PROJECT_NAME} PRIVATE core mmstd_datatools Eigen bhtsne Delau)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
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
endif()
