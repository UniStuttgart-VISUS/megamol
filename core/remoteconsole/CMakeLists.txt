#
# MegaMol™ Remote Console
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

option(BUILD_REMOTECONSOLE "Build remote Lua console" ON)

if(BUILD_REMOTECONSOLE)
  project(remoteconsole)

  # include externals
  include(ExternalProject)

  if(UNIX)
    set(CXXOPTS_INSTALL_CMD "eval")
    set(CXXOPTS_INSTALL_ARGS "cp ${PROJECT_BINARY_DIR}/extlibcxxopts-prefix/src/extlibcxxopts/include/*.hpp ${EXTERNAL_INSTALL_LOCATION}/include/")
  else()
    set(CXXOPTS_INSTALL_CMD "powershell")
    set(CXXOPTS_INSTALL_ARGS "copy ${PROJECT_BINARY_DIR}/extlibcxxopts-prefix/src/extlibcxxopts/include/*.hpp ${EXTERNAL_INSTALL_LOCATION}/include/")
  endif()
  set(CXXOPTS_VER "1_4_4")
  string(REPLACE "_" "." CXXOPTS_TAG ${CXXOPTS_VER})
  string(CONCAT CXXOPTS_TAG "v" ${CXXOPTS_TAG})
  ExternalProject_Add(extlibcxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG ${CXXOPTS_TAG}
    DEPENDS extlibzmq core
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CXXOPTS_INSTALL_CMD} ${CXXOPTS_INSTALL_ARGS}
    INSTALL_COMMAND ""
  )
  message(STATUS "cxxopts Git Tag: " ${CXXOPTS_TAG})

  set(LIBS lua ${ZMQ_LIB})
  include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
  link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

  # Build MegaMol Remote Console
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  include_directories(BEFORE "include" "src")

  # Target definition
  add_executable(${PROJECT_NAME} ${header_files} ${source_files})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  add_dependencies(${PROJECT_NAME} extlibcxxopts extlibzmq core)

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

  # Installation rules for generated files
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")

endif(BUILD_REMOTECONSOLE)

