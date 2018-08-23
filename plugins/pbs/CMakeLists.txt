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
    set(SNAPPY_IMPORT_LIB "lib/snappy.lib")
    set(SNAPPY_LIB "bin/snappy.dll")
    set(ICET_IMPORT_LIB "lib/IceTCore.lib" "lib/IceTGL.lib" "lib/IceTMPI.lib")
    set(ICET_LIB "bin/IceTCore.dll" "bin/IceTGL.dll" "bin/IceTMPI.dll")
  else()
    include(GNUInstallDirs)
    set(SNAPPY_LIB "${CMAKE_INSTALL_LIBDIR}/libsnappy.so")
    set(ICET_LIB "lib/libIceTCore.so" "lib/libIceTGL.so" "lib/libIceTMPI.so")
  endif()
  
  add_external_library(snappy SHARED
    GIT_REPOSITORY https://github.com/google/snappy.git
    GIT_TAG ${SNAPPY_VER}
	IMPORT_LIBRARY_DEBUG ${SNAPPY_IMPORT_LIB}
    IMPORT_LIBRARY_RELEASE ${SNAPPY_IMPORT_LIB}
    LIBRARY_DEBUG ${SNAPPY_LIB}
    LIBRARY_RELEASE ${SNAPPY_LIB}
    CMAKE_ARGS 
      -DBUILD_SHARED_LIBS=ON 
      -DSNAPPY_BUILD_TESTS=OFF)

  add_external_library(icet SHARED
    GIT_REPOSITORY https://gitlab.kitware.com/icet/icet.git
	IMPORT_LIBRARY_DEBUG ${ICET_IMPORT_LIB}
    IMPORT_LIBRARY_RELEASE ${ICET_IMPORT_LIB}
    LIBRARY_DEBUG ${ICET_LIB}
    LIBRARY_RELEASE ${ICET_LIB}
    CMAKE_ARGS 
      -DBUILD_SHARED_LIBS=ON
      -DICET_BUILD_TESTING=OFF
      -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "resources/*")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES
    SUFFIX ".mmplg"
    CXX_STANDARD 17)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} 
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PUBLIC "include" "src")
  target_link_libraries(${PROJECT_NAME} PRIVATE vislib core glad OSPRay_plugin image_calls zfp libzmq snappy icet)

  if(MPI_C_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_C_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})
  endif()

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    install(DIRECTORY ${SNAPPY_DLLDIR} DESTINATION ${CMAKE_INSTALL_PREFIX} FILES_MATCHING PATTERN "snappy*.dll")
    install(DIRECTORY ${ICET_DLLDIR} DESTINATION ${CMAKE_INSTALL_PREFIX} FILES_MATCHING PATTERN "IceT*.dll")
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
  else()
    file(GLOB SNAPPY_LIB_FILES "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libsnappy.so*")
    install(FILES ${SNAPPY_LIB_FILES} DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
    file(GLOB ICET_LIB_FILES "${EXTERNAL_INSTALL_LOCATION}/lib/libIceTCore.so*" "${EXTERNAL_INSTALL_LOCATION}/lib/libIceTGL.so*" "${EXTERNAL_INSTALL_LOCATION}/lib/libIceTMPI.so*")
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
