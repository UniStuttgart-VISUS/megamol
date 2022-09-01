# MegaMol
# Copyright (c) 2017, MegaMol Dev Team
# All rights reserved.
#

option(BUILD_REMOTECONSOLE "Build remote Lua console" ON)

if(BUILD_REMOTECONSOLE)
  project(remoteconsole)

  find_package(ZeroMQ REQUIRED)
  find_package(cppzmq REQUIRED)
  find_package(cxxopts)

  # Glob source files.
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")

  # Setup target.
  add_executable(${PROJECT_NAME} ${header_files} ${source_files})
  target_include_directories(${PROJECT_NAME} 
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      libzmq
      cppzmq
      cxxopts::cxxopts)

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

  # Installation rules for generated files
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
endif(BUILD_REMOTECONSOLE)
