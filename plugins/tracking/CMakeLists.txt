#
# MegaMol™ tracking Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER tracking EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build tracking" ON)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(tracking)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  
  # Get the external vrpn from GitHub.
  include(ExternalProject)
  
  set(VRPN_INSTALL_LOCATION ${CMAKE_SOURCE_DIR}/external CACHE INTERNAL '')
  set(VRPN_VER "v07.34")
  
  set(VRPN_INSTALL_COPY debug "-E copy ${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn-build/quat/Debug/quatd.lib ${VRPN_INSTALL_LOCATION}/lib" optimized "-E copy ${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn-build/quat/Release/quat.lib ${VRPN_INSTALL_LOCATION}/lib")
  ExternalProject_Add(
	vrpn
    GIT_REPOSITORY https://github.com/vrpn/vrpn.git
    GIT_TAG ${VRPN_VER} # the only version the software was tested with
	GIT_SUBMODULES
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ${CMAKE_COMMAND} -DVRPN_BUILD_CLIENTS:BOOL=OFF -DVRPN_BUILD_SERVERS:BOOL=OFF ${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn COMMAND ${CMAKE_COMMAND} --build "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn-build"
	INSTALL_COMMAND ${CMAKE_COMMAND} ${VRPN_INSTALL_COPY}
  )

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE" CACHE INTERNAL "")
  option(USE_NATNET "use NatNet" ON)
  if(USE_NATNET)
    find_package(NATNET REQUIRED)
    message(STATUS ${NATNET_LIBRARIES})
    message(STATUS ${NATNET_INCLUDE_DIR})
    if (!NATNET_FOUND)
     message(FATAL_ERROR "Cannot find NatNet")
    endif()
  endif(USE_NATNET)
  
  set(LIBS vislib core glad ${NATNET_LIBRARIES})
  include_directories(${NATNET_INCLUDE_DIR} ${vrpn_install_dir}/include "include" "src")

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  #file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  add_dependencies(${PROJECT_NAME} vislib core vrpn)
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
  #target_link_libraries(${PROJECT_NAME} PRIVATE ${NATNET})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
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
  #source_group("Shaders" FILES ${shader_files})


endif(BUILD_${EXPORT_NAME}_PLUGIN)