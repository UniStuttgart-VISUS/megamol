cmake_minimum_required(VERSION 3.12)

project(bhtsne CXX)

set(public_header_files
  tsne.h)
set(header_files
  sptree.h
  vptree.h)
set(source_files
  tsne.cpp
  sptree.cpp)

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)

add_library(bhtsne ${public_header_files} ${header_files} ${source_files})
set_target_properties(bhtsne PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bhtsne
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

include(GenerateExportHeader)
generate_export_header(bhtsne)

include(GNUInstallDirs)

install(FILES
  ${public_header_files}
  "${CMAKE_CURRENT_BINARY_DIR}/bhtsne_export.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(TARGETS bhtsne
  EXPORT bhtsneTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT bhtsneTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bhtsne
  NAMESPACE bhtsne::
  FILE bhtsneConfig.cmake)
