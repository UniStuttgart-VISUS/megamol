#
# MegaMol™ vtkm Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER mmvtkm EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build mmvtkm" OFF)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(mmvtkm)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE" CACHE INTERNAL "") #TODO add ENABLE_CUDA if intended to run vtkm with cuda and cuda needs to be enabled?
  
  # Add externals.
  require_external(vtkm)

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  #file(GLOB_RECURSE library_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "lib/*") # comment this line if implemented as external library
  #file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    "include" "src" "lib"
    # TODO set include dirs with targets in cmakeexternals and not here
    "${CMAKE_CURRENT_BINARY_DIR}/vtkm_ext-prefix/src/vtkm_ext"
    "${CMAKE_CURRENT_BINARY_DIR}/vtkm_ext-prefix/src/vtkm_ext-build/include"
    "${CMAKE_CURRENT_BINARY_DIR}/vtkm_ext-prefix/src/vtkm_ext/vtkm/thirdparty/taotuple"
    "${CMAKE_CURRENT_BINARY_DIR}/vtkm_ext-prefix/src/vtkm_ext/vtkm/thirdparty/diy")
  target_link_libraries(${PROJECT_NAME} PRIVATE core vtkm_cont vtkm_renderer vtkm_worklet)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  # install_external(TARGETS vtkm_cont)
  # install_external(TARGETS vtkm_renderer)
  # install_external(TARGETS vtkm_worklet)
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    if(MSVC)
      install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
    endif()
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  #source_group("Shaders" FILES ${shader_files})
endif()
