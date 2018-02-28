#
# MegaMol™ tracking Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

# Stop if the user tries to build this plugin with Linux.
if(WIN32)
  
  string(TOUPPER tracking EXPORT_NAME)
  option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build tracking" ON)
  
  if(BUILD_${EXPORT_NAME}_PLUGIN)  
    project(tracking)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    
    # Get the external vrpn project from GitHub.
    include(ExternalProject)
    
    # Set the install location and the version for the vrpn project.
    set(VRPN_INSTALL_LOCATION ${CMAKE_SOURCE_DIR}/external CACHE INTERNAL '')
    set(VRPN_VER "v07.34")
    
    # Set the paths for the needed libraries and include files from the vrpn
    # project.
    set(VRPN_INCLUDE_COPY "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Assert.h" "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_BaseClass.h" 
                          "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Button.h" "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Configure.h" 
  						  "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Connection.h" "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_EndpointContainer.h" 
  						  "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Serial.h" "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Shared.h" 
  						  "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Thread.h" "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Tracker.h" 
  						  "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_Types.h" "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn/vrpn_WindowsH.h")
    
    # Add the vrpn project as an external project and set the cmake commands
    # for configuring and building it. For some reason the x64 flag is
    # necessary because if it is not set vrpn is using x86.
    ExternalProject_Add(
  	  vrpn
      GIT_REPOSITORY https://github.com/vrpn/vrpn.git
      GIT_TAG ${VRPN_VER} # the only version the software was tested with
  	  GIT_SUBMODULES
  	  CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_GENERATOR_PLATFORM=x64 -DVRPN_BUILD_CLIENTS:BOOL=OFF -DVRPN_BUILD_SERVERS:BOOL=OFF ${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn
  	  BUILD_COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn-build
  	  INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${VRPN_INCLUDE_COPY} ${VRPN_INSTALL_LOCATION}/include
    )
    
    # Set the paths to the libraries.
    set(QUAT_LIBRARY debug "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn-build/quat/Debug/quatd.lib" optimized "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn-build/quat/Release/quat.lib")
    set(VRPN_LIBRARY debug "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn-build/Debug/vrpnd.lib" optimized "${PROJECT_BINARY_DIR}/vrpn-prefix/src/vrpn-build/Release/vrpn.lib")  
    
    # Check option dependencies: First list item is always the current build option,
    # afterwards list the dependencies.
    set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE" CACHE INTERNAL "")
    
    # Find the NatNet library, the location of the NatNet dll and the location of the
    # inlucde files.
    option(USE_NATNET "use NatNet" ON)
    if(USE_NATNET)
      find_package(NATNET REQUIRED)
      message(STATUS ${NATNET_LIBRARIES})
      message(STATUS ${NATNET_DLL_DIR})
      message(STATUS ${NATNET_INCLUDE_DIR})
      if (NOT NATNET_FOUND)
        message(FATAL_ERROR "Cannot find NatNet")
      endif()
    endif(USE_NATNET)
    
    set(LIBS vislib core glad ${NATNET_LIBRARIES} ${VRPN_LIBRARY} ${QUAT_LIBRARY})
    include_directories(${NATNET_INCLUDE_DIR} ${VRPN_INSTALL_LOCATION}/include "include" "src")
  
    file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
    file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
    file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  
    # Target definition
    add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
    add_dependencies(${PROJECT_NAME} vislib core vrpn)
    # Set target naming conventions for different build types
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
    # Dependent projects only need to link against the core.so itself.
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
    target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  
    # Installation rules for generated files
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
    if(WIN32)
      install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  	  install(FILES ${NATNET_DLL_DIR} DESTINATION "bin")
      install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    else()
      install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
    endif()
  
    # Grouping in Visual Studio
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
    source_group("Public Header Files" FILES ${public_header_files})
    source_group("Header Files" FILES ${header_files})
    source_group("Source Files" FILES ${source_files})
  
  
  endif(BUILD_${EXPORT_NAME}_PLUGIN)

endif(WIN32)