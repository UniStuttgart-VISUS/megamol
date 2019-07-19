#
# MegaMol™ mesh Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER mesh EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build mesh" ON)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(mesh)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  if(UNIX)
    set(quickhull_cmake_args -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS="-fPIC")
  else()
    set(quickhull_cmake_args "")
  endif()

  add_external_project(quickhull_ext
    GIT_REPOSITORY https://github.com/akuukka/quickhull.git
    #GIT_TAG 
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/quickhull/CMakeLists.txt"
    "<SOURCE_DIR>/CMakeLists.txt"
    CMAKE_ARGS ${quickhull_cmake_args}
  )
  if(WIN32)
  set(QUICKHULL_IMPORT_LIB "lib/quickhull.lib")
  set(QUICKHULL_IMPORT_LIB_DEBUG ${QUICKHULL_IMPORT_LIB})
  set(QUICKHULL_LIB "bin/quickhull.dll")
  set(QUICKHULL_LIB_DEBUG ${QUICKHULL_LIB})
else()
  include(GNUInstallDirs)
  set(QUICKHULL_IMPORT_LIB_DEBUG "${CMAKE_INSTALL_LIBDIR}/libquickhull.so")
  set(QUICKHULL_IMPORT_LIB "${CMAKE_INSTALL_LIBDIR}/libquickhull.so")
  set(QUICKHULL_LIB_DEBUG "${CMAKE_INSTALL_LIBDIR}/libquickhull.so")
  set(QUICKHULL_LIB "${CMAKE_INSTALL_LIBDIR}/libquickhull.so")
endif()
  add_external_library(quickhull SHARED
    DEPENDS quickhull_ext
    INCLUDE_DIR "include"
    IMPORT_LIBRARY_DEBUG ${QUICKHULL_IMPORT_LIB_DEBUG}
    IMPORT_LIBRARY_RELEASE ${QUICKHULL_IMPORT_LIB}
    LIBRARY_DEBUG ${QUICKHULL_LIB_DEBUG}
    LIBRARY_RELEASE ${QUICKHULL_LIB})
  get_target_property(quickhull_include quickhull INTERFACE_INCLUDE_DIRECTORIES)

  message(STATUS "INCLUDE: ${quickhull_include}")

  # Build MegaMol Plugin
  # public header files for installation
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  # source files for compilation
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  add_dependencies(${PROJECT_NAME} vislib core quickhull)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg" CXX_STANDARD 17)
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE core PUBLIC quickhull)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
                                                    "src"
                                                    $<BUILD_INTERFACE:${quickhull_include}>)

  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)

  # Additional definitions for compiling with VS
  if(MSVC)
      add_definitions(-DNOMINMAX)
  endif()

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  install_external(TARGETS quickhull)
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Shaders" FILES ${shader_files})


endif(BUILD_${EXPORT_NAME}_PLUGIN)