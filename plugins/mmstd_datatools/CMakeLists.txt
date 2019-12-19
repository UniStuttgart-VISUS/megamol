#
# MegaMol™ mmstd_datatools Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER mmstd_datatools EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build mmstd_datatools" ON)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(mmstd_datatools)
  
  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_GEOMETRY_CALLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  # Add externals.
  require_external(tinyply)
  require_external(mmpld_io)
  require_external(Eigen)
  require_external(nanoflann)

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp" "3rd/min_sphere_of_spheres/*.cpp")
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

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src"
    "3rd/min_sphere_of_spheres")
  target_link_libraries(${PROJECT_NAME} PRIVATE core geometry_calls tinyply Eigen nanoflann mmpld_io)
  if(MPI_C_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_C)
  endif()

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
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
endif()
