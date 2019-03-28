#
# MegaMol™ OSPRay_plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_OSPRAY_PLUGIN "Option to build OSPRay_plugin" OFF)

if(BUILD_OSPRAY_PLUGIN)
  project(OSPRay_plugin)

  string(TOUPPER ${PROJECT_NAME} EXPORT_NAME)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_OSPRAY_PLUGIN BUILD_CORE BUILD_GEOMETRY_CALLS_PLUGIN BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_MMSTD_TRISOUP_PLUGIN BUILD_PROTEIN_CALLS_PLUGIN" CACHE INTERNAL "")

  find_package(ospray CONFIG REQUIRED)

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src" ${OSPRAY_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE core geometry_calls mmstd_datatools mmstd_trisoup protein_calls ${OSPRAY_LIBRARIES})

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    install(DIRECTORY "${OSPRAY_ROOT}/bin/" DESTINATION "bin" FILES_MATCHING PATTERN "*.dll")
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
    if (TARGET ospray::ospray)
        get_target_property(OSPRAY_LIB_PATH ospray::ospray IMPORTED_LOCATION_RELEASE)
        get_filename_component(dir ${OSPRAY_LIB_PATH} DIRECTORY)
        install(DIRECTORY "${dir}" DESTINATION "${CMAKE_INSTALL_PREFIX}" FILES_MATCHING PATTERN "lib*")
    else(TARGET ospray::ospray)
        foreach(lib ${OSPRAY_LIBRARIES})
            get_filename_component(dir ${lib} DIRECTORY)
            install(DIRECTORY "${dir}" DESTINATION "${CMAKE_INSTALL_PREFIX}" FILES_MATCHING PATTERN "lib*")
        endforeach()
    endif(TARGET ospray::ospray)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Shaders" FILES ${shader_files})
endif()
