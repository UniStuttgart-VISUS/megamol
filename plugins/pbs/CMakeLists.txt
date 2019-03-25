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

  if(WIN32)
    set(SNAPPY_IMPORT_LIB "lib/snappy.lib")
    set(SNAPPY_LIB "bin/snappy.dll")
    set(ICET_CORE_IMPORT_LIB "lib/IceTCore.lib")
    set(ICET_CORE_LIB "bin/IceTCore.dll")
    set(ICET_GL_IMPORT_LIB "lib/IceTGL.lib")
    set(ICET_GL_LIB "bin/IceTGL.dll")
    set(ICET_MPI_IMPORT_LIB "lib/IceTMPI.lib")
    set(ICET_MPI_LIB "bin/IceTMPI.dll")
  else()
    include(GNUInstallDirs)
    set(SNAPPY_LIB "${CMAKE_INSTALL_LIBDIR}/libsnappy.so")
    set(ICET_CORE_LIB "lib/libIceTCore.so")
    set(ICET_GL_LIB "lib/libIceTGL.so")
    set(ICET_MPI_LIB "lib/libIceTMPI.so")
  endif()

  add_external_project(snappy_ext
    GIT_REPOSITORY https://github.com/google/snappy.git
    GIT_TAG "1.1.7"
    CMAKE_ARGS
      -DBUILD_SHARED_LIBS=ON
      -DSNAPPY_BUILD_TESTS=OFF)
  add_external_library(snappy SHARED
    DEPENDS snappy_ext
    IMPORT_LIBRARY_DEBUG ${SNAPPY_IMPORT_LIB}
    IMPORT_LIBRARY_RELEASE ${SNAPPY_IMPORT_LIB}
    LIBRARY_DEBUG ${SNAPPY_LIB}
    LIBRARY_RELEASE ${SNAPPY_LIB})

  if(MPI_C_FOUND)
    add_external_project(IceT_ext
      GIT_REPOSITORY https://gitlab.kitware.com/icet/icet.git
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DICET_BUILD_TESTING=OFF
        -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})
    add_external_library(IceTCore SHARED
      DEPENDS IceT_ext
      IMPORT_LIBRARY_DEBUG ${ICET_CORE_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${ICET_CORE_IMPORT_LIB}
      LIBRARY_DEBUG ${ICET_CORE_LIB}
      LIBRARY_RELEASE ${ICET_CORE_LIB})
    add_external_library(IceTGL SHARED
      DEPENDS IceT_ext
      IMPORT_LIBRARY_DEBUG ${ICET_GL_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${ICET_GL_IMPORT_LIB}
      LIBRARY_DEBUG ${ICET_GL_LIB}
      LIBRARY_RELEASE ${ICET_GL_LIB})
    add_external_library(IceTMPI SHARED
      DEPENDS IceT_ext
      IMPORT_LIBRARY_DEBUG ${ICET_MPI_IMPORT_LIB}
      IMPORT_LIBRARY_RELEASE ${ICET_MPI_IMPORT_LIB}
      LIBRARY_DEBUG ${ICET_MPI_LIB}
      LIBRARY_RELEASE ${ICET_MPI_LIB})
  endif()

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
  target_link_libraries(${PROJECT_NAME} PRIVATE core OSPRay_plugin image_calls zfp libzmq libcppzmq snappy)
  if(MPI_C_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE IceTCore IceTGL IceTMPI)
  endif()

  if(MPI_C_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_C_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})
  endif()

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")
  install_external(TARGETS snappy)
  if(MPI_C_FOUND)
    install_external(TARGETS IceTCore IceTGL IceTMPI)
  endif()
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Resources" FILES ${resource_files})
endif()
