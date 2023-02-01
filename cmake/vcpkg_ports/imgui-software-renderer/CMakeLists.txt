cmake_minimum_required(VERSION 3.16)

project(imgui_software_renderer
  LANGUAGES CXX)

find_package(imgui CONFIG REQUIRED)

add_library(${PROJECT_NAME} STATIC)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/imgui_software_renderer>)
target_link_libraries(${PROJECT_NAME} PRIVATE imgui::imgui)
target_sources(${PROJECT_NAME} PRIVATE src/imgui_sw.cpp)

include(GNUInstallDirs)

install(FILES
  src/imgui_sw.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/imgui_software_renderer")

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT ${PROJECT_NAME}Targets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_NAME}Config.cmake)
