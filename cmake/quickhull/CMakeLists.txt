cmake_minimum_required(VERSION 3.10)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
project(quickhull)

option(BUILD_TESTS "Option to build tests" OFF)
if(BUILD_TESTS)
  add_subdirectory(Tests)
endif()

file(GLOB struct_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Structs/*.hpp")
file(GLOB source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
file(GLOB header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.hpp")

add_library(${PROJECT_NAME} SHARED ${struct_files} ${header_files} ${source_files})
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Structs>)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Structs/ DESTINATION "include/Structs")
install(FILES ${header_files} DESTINATION "include")
if(WIN32)
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
else()
  install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
endif()
