#
# MegaMol™ probe Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER probe EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build probe" OFF)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(probe)

  #find_package(Qhull REQUIRED)

  if(WIN32)
    set(QHULL_IMPORT_LIB "lib/qhull.lib")
    set(QHULL_LIB "bin/qhull.dll")
  else()
    include(GNUInstallDirs)
    set(QUHULL_IMPORT_LIB "lib/libqhull.so")
    set(QUHULL_LIB "lib/libqhull.so")
  endif()

  add_external_project(qhull STATIC
    GIT_REPOSITORY https://github.com/qhull/qhull.git
    GIT_TAG "v7.3.2"
    BUILD_BYPRODUCTS "<INSTALL_DIR>/${QHULL_IMPORT_LIB}" "<INSTALL_DIR>/${QHULL_LIB}"
    #CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    #CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/qhull/CMakeLists.txt"
      "<SOURCE_DIR>/CMakeLists.txt")
  add_external_library(qhull
    INCLUDE_DIR "include"
    IMPORT_LIBRARY ${QHULL_IMPORT_LIB}
    LIBRARY ${QHULL_IMPORT_LIB})

  require_external(Eigen)
  require_external(nanoflann)
  require_external(libigl)

  find_package(CGAL CONFIG REQUIRED)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_ADIOS_PLUGIN_PLUGIN BUILD_MESH_PLUGIN BUILD_MMSTD_DATATOOLS_PLUGIN" CACHE INTERNAL "")

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE third RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/*")
  file(GLOB_RECURSE ospray_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "probe_ospray/src/*.cpp")
  file(GLOB_RECURSE ospray_headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "probe_ospray/src/*.h")
  file(GLOB_RECURSE ospray_public_headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "probe_ospray/include/*.h")

  

  # Target definition
  #add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${shader_files} ${source_files})
  if(BUILD_OSPRAY_PLUGIN_PLUGIN)
    add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files} ${third} ${ospray_sources} ${ospray_headers} ${ospray_public_headers})
  else()
    add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files} ${third})
  endif()
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  if(BUILD_OSPRAY_PLUGIN_PLUGIN)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_HAS_OSPRAY)
  endif()

  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    "include" "src" "3rd")

  if(BUILD_OSPRAY_PLUGIN_PLUGIN)
    target_include_directories(${PROJECT_NAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/probe_ospray/include> "probe_ospray/src")
  endif()

  find_library(blend2d blend2d REQUIRED)
  find_path(blend2d_INCLUDES blend2d.h)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${blend2d})
  target_include_directories(${PROJECT_NAME} PRIVATE ${blend2d_INCLUDES})

  target_link_libraries(${PROJECT_NAME} PRIVATE core adios_plugin Eigen nanoflann mesh qhull libigl mmstd_datatools CGAL::CGAL)# Qhull::libqhull )
  if(BUILD_OSPRAY_PLUGIN_PLUGIN)
    target_link_libraries(${PROJECT_NAME} PUBLIC OSPRay_plugin)
  endif()

  # Installation rules for generated files
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")

  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    #install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    #install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    #install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("3rd Files" FILES ${third})
  #source_group("Shaders" FILES ${shader_files})

  # Register plugin
  megamol_register_plugin(${PROJECT_NAME})
endif()
