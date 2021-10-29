#
# MegaMol™ Core
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_CORE "Build core" ON)

if(BUILD_CORE)
  project(core)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_CORE BUILD_VISLIB" CACHE INTERNAL "")

  # Add externals.
  require_external(expat)
  require_external(lua)
  require_external(libzmq)
  require_external(libcppzmq)
  require_external(zlib)
  require_external(libpng)
  require_external(glm)
  require_external(json)

  if (ENABLE_CUESDK)
    require_external(CUESDK)
  endif()

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h" "include/*.inl")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Resources/*")
  set(version_files ${CMAKE_BINARY_DIR}/version/MMCoreVersionInfo.cpp)

  # Target definition
  add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files} ${version_files})
  target_compile_definitions(${PROJECT_NAME} PRIVATE MEGAMOLCORE_EXPORTS)
  if (ENABLE_GL)
  target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "src" "include" "${MEGAMOL_VISLIB_GL_DIR}/include" "${MEGAMOL_CORE_GL_DIR}/include")
  else()
  target_include_directories(${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "src" "include")
  endif()
  target_link_libraries(${PROJECT_NAME} 
    PUBLIC vislib glm zlib libpng json lua expat frontend_resources libcppzmq libzmq
    PRIVATE ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} abstract_frontend_service) # abstract render api used by MegaMolGraph

  if(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
  endif()

  if (ENABLE_CUESDK)
    target_link_libraries(${PROJECT_NAME} PUBLIC CUESDK)
  endif()

  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
      LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DYNAMICBASE:NO")
  endif()

  if(MPI_C_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_C)
  endif()

  # Source groups in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
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
    string(REGEX REPLACE "^include\\\\mmcore" "Public Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  source_group("Resources" FILES ${resource_files})
  source_group("Version" FILES ${version_files})

  # Installation rules for generated files
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Resources/ DESTINATION "share/resources")
  
endif(BUILD_CORE)
