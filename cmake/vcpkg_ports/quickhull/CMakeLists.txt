cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(quickhull CXX)

option(BUILD_TESTS "Option to build tests" OFF)
if(BUILD_TESTS)
  add_subdirectory(Tests)
endif()

file(GLOB struct_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Structs/*.hpp")
file(GLOB header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.hpp")
file(GLOB source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")

add_library(${PROJECT_NAME} ${struct_files} ${header_files} ${source_files})

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Structs>)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME})
  
include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Structs/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Structs")
install(FILES ${header_files} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(TARGETS ${PROJECT_NAME}
  EXPORT "${PROJECT_NAME}Targets"
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT "${PROJECT_NAME}Targets"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_NAME}Config.cmake)

