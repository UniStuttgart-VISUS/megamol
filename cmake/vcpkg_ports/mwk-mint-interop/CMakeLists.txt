cmake_minimum_required(VERSION 3.13)

project(mwk-mint-interop
  VERSION 1.0
  LANGUAGES CXX)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(spout2 CONFIG REQUIRED)

add_library(mwk-mint-interop STATIC
  src/interop.cpp
  include/interop.hpp)

target_include_directories(mwk-mint-interop
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(mwk-mint-interop PRIVATE cxx_std_17)

target_link_libraries(mwk-mint-interop
  PRIVATE
    nlohmann_json::nlohmann_json
    libzmq
    cppzmq
    Spout2::Spout)

# Install
include(GNUInstallDirs)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS mwk-mint-interop
  EXPORT mwk-mint-interopTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT mwk-mint-interopTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mwk-mint-interop
  NAMESPACE mwk-mint-interop::
  FILE mwk-mint-interopConfigTargets.cmake)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/mwk-mint-interopConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/mwk-mint-interopConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mwk-mint-interop")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/mwk-mint-interopConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/mwk-mint-interopConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/mwk-mint-interopConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mwk-mint-interop")
