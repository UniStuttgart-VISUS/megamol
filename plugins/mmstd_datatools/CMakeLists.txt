#
# MegaMol™ mmstd_datatools Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_MMSTD_DATATOOLS_PLUGIN "Option to build mmstd_datatools" ON)

if(BUILD_MMSTD_DATATOOLS_PLUGIN)
  project(mmstd_datatools)
  
  string(TOUPPER ${PROJECT_NAME} EXPORT_NAME)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_GEOMETRY_CALLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  if (OPENMP_FOUND OR WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()

  add_external_project(tinyply_ext
  GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
  GIT_TAG "2.1"
  CMAKE_ARGS -DSHARED_LIB=true)

  if(WIN32)
    set(TNY_IMPORT_LIB "lib/tinyply.lib")
    set(TNY_IMPORT_LIB_DEBUG "lib/tinyplyd.lib")
    set(TNY_LIB "bin/tinyply.dll")
    set(TNY_LIB_DEBUG "bin/tinyplyd.dll")
  else()
    include(GNUInstallDirs)
    set(TNY_IMPORT_LIB_DEBUG "lib/libtinyply.so")
    set(TNY_IMPORT_LIB "lib/libtinyply.so")
    set(TNY_LIB_DEBUG "lib/libtinyply.so")
    set(TNY_LIB "lib/libtinyply.so")
  endif()

  add_external_library(tinyply SHARED
  DEPENDS tinyply_ext
  IMPORT_LIBRARY_DEBUG ${TNY_IMPORT_LIB_DEBUG}
  IMPORT_LIBRARY_RELEASE ${TNY_IMPORT_LIB}
  LIBRARY_DEBUG ${TNY_LIB_DEBUG}
  LIBRARY_RELEASE ${TNY_LIB})

  # Add Eigen.
  add_external_project(Eigen_ext
    GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
    GIT_TAG "3.3.4"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND "")
  add_external_library(Eigen INTERFACE
    DEPENDS Eigen_ext
    INCLUDE_DIR "src/Eigen_ext")

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

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src"
    "3rd/nanoflann")
  target_link_libraries(${PROJECT_NAME} PRIVATE core geometry_calls tinyply Eigen)

  if(MPI_C_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_C_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})
  endif()

  if (OPENMP_FOUND OR WIN32)
    set_target_properties(${PROJECT_NAME}
      PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install_external(TARGETS tinyply)
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
endif()
