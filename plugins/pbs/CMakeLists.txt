#
# MegaMol™ pbs Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER pbs EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build pbs" OFF)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(pbs)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_OSPRAY_PLUGIN" CACHE INTERNAL "")

  set(SNAPPY_VER "1.1.7")
  if (MSVC)
    set(SNAPPY_LIB "${EXTERNAL_INSTALL_LOCATION}/lib/snappy.lib" CACHE INTERNAL '')
    set(SNAPPY_DLLDIR "${EXTERNAL_INSTALL_LOCATION}/bin")
    set(ICET_LIB "${EXTERNAL_INSTALL_LOCATION}/lib/IceTCore.lib" "${EXTERNAL_INSTALL_LOCATION}/lib/IceTGL.lib" "${EXTERNAL_INSTALL_LOCATION}/lib/IceTMPI.lib" CACHE INTERNAL '')
    set(ICET_DLLDIR "${EXTERNAL_INSTALL_LOCATION}/bin")
  else()
    include(GNUInstallDirs)
    set(SNAPPY_LIB "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libsnappy.so")
    set(ICET_LIB "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libIceTCore.so" "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libIceTGL.so" "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libIceTMPI.so")
  endif()
  
  Include(ExternalProject)
  ExternalProject_Add(extsnappy
    GIT_REPOSITORY https://github.com/google/snappy.git
    GIT_TAG ${SNAPPY_VER}
    CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DSNAPPY_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
  )
  message(STATUS "SNAPPY Git Tag: " ${SNAPPY_VER})

  ExternalProject_Add(exticet
    GIT_REPOSITORY https://gitlab.kitware.com/icet/icet.git
    CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DICET_BUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
  )

  set(LIBS vislib core glad OSPRay_plugin ${ZFP_LIB} ${ZMQ_LIB} ${SNAPPY_LIB} ${ICET_LIB})
  include_directories(${EXTERNAL_INSTALL_LOCATION}/include "include" "src")
  link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

  if (MPI_C_FOUND)
    include_directories(${MPI_C_INCLUDE_PATH})
    set(LIBS ${LIBS} ${MPI_C_LIBRARIES})
  endif()

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "resources/*")

  # Target definition
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
  add_dependencies(${PROJECT_NAME} vislib core)
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    install(DIRECTORY ${SNAPPY_DLLDIR} DESTINATION ${CMAKE_INSTALL_PREFIX} FILES_MATCHING PATTERN "snappy*.dll")
    install(DIRECTORY ${ICET_DLLDIR} DESTINATION ${CMAKE_INSTALL_PREFIX} FILES_MATCHING PATTERN "IceT*.dll")
  else()
    file(GLOB SNAPPY_LIB_FILES "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libsnappy.so*")
    install(FILES ${SNAPPY_LIB_FILES} DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
    file(GLOB ICET_LIB_FILES "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libIceT*.so*")
    install(FILES ${ICET_LIB_FILES} DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Resources" FILES ${resource_files})


endif(BUILD_${EXPORT_NAME}_PLUGIN)