#
# MegaMol™ Core
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_CORE "Build core" ON)

if (BUILD_CORE)
  project(core)

  # Add externals.
  find_package(nlohmann_json CONFIG REQUIRED)
  find_package(PNG REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(ZeroMQ REQUIRED)
  find_package(cppzmq REQUIRED)
  find_package(glm REQUIRED)
  find_package(TBB CONFIG REQUIRED)

  require_external(expat)
  require_external(lua)
  
  if (ENABLE_CUESDK)
    require_external(CUESDK)
  endif ()

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h" "include/*.inl")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")

  # Target definition
  add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files})
  target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      vislib
      glm::glm
      ZLIB::ZLIB
      PNG::PNG
      nlohmann_json::nlohmann_json
      lua
      expat
      frontend_resources
      libzmq
      cppzmq
      OpenMP::OpenMP_CXX
      TBB::tbb
    PRIVATE
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
      abstract_frontend_service # abstract render api used by MegaMolGraph
      megamol_build_info_rc
    )
  if (UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
  endif ()
  if (WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE Version)
  endif ()
  if (ENABLE_CUESDK)
    target_link_libraries(${PROJECT_NAME} PUBLIC CUESDK)
  endif ()

  if (MPI_C_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_C)
  endif ()

  # Install resources
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")

  # Source groups in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include/mmcore" PREFIX "Public Header Files" FILES ${public_header_files})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Header Files" FILES ${header_files})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${source_files})

endif (BUILD_CORE)
