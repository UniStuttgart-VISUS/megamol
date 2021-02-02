cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(lua
  LANGUAGES C)

file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c")
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
set(public_header_files lua.h lualib.h lauxlib.h luaconf.h lua.hpp)

add_library(${PROJECT_NAME} STATIC ${source_files} ${header_files})
target_include_directories(${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS ${PROJECT_NAME} DESTINATION "lib")
install(FILES ${public_header_files} DESTINATION "include")
