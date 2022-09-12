cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(imguizmoquat
  VERSION 3.0
  LANGUAGES CXX)

find_package(imgui REQUIRED)

set(IMGUIZMOQUAT_HEADERS
  imGuIZMO.quat/imGuIZMOquat.h
  imGuIZMO.quat/vgConfig.h
  imGuIZMO.quat/vGizmo.h
  imGuIZMO.quat/vGizmoMath.h
  imGuIZMO.quat/vgMath.h)
set(IMGUIZMOQUAT_SOURCES
  imGuIZMO.quat/imGuIZMOquat.cpp)

add_library(imguizmoquat STATIC ${IMGUIZMOQUAT_SOURCES} ${IMGUIZMOQUAT_HEADERS})
set_target_properties(imguizmoquat PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(imguizmoquat PRIVATE IMGUIZMO_IMGUI_FOLDER=./)
target_include_directories(imguizmoquat PRIVATE imGuIZMO.quat)
target_link_libraries(imguizmoquat PRIVATE imgui::imgui)

# Install
include(GNUInstallDirs)

install(FILES ${IMGUIZMOQUAT_HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS imguizmoquat
  EXPORT imguizmoquatTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT imguizmoquatTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/imguizmoquat
  NAMESPACE imguizmoquat::
  FILE imguizmoquatConfigTargets.cmake)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/imguizmoquatConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/imguizmoquatConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/imguizmoquat")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/imguizmoquatConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/imguizmoquatConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/imguizmoquatConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/imguizmoquat")
