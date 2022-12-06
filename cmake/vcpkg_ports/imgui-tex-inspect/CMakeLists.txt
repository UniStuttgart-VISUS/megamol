cmake_minimum_required(VERSION 3.16)

project(imgui_tex_inspect
  LANGUAGES CXX)

option(IMGUI_TEX_INSPECT_USE_OPENGL3 "" OFF)

find_package(imgui CONFIG REQUIRED)

add_library(${PROJECT_NAME} STATIC)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backends>
    $<INSTALL_INTERFACE:include/imgui_tex_inspect>)
target_link_libraries(${PROJECT_NAME} PRIVATE imgui::imgui)
target_sources(${PROJECT_NAME} PRIVATE imgui_tex_inspect.cpp)

if (IMGUI_TEX_INSPECT_USE_OPENGL3)
  target_sources(${PROJECT_NAME} PRIVATE backends/tex_inspect_opengl.cpp)
endif ()

include(GNUInstallDirs)

install(FILES
  imgui_tex_inspect.h
  imgui_tex_inspect_internal.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/imgui_tex_inspect")

if (IMGUI_TEX_INSPECT_USE_OPENGL3)
  install(FILES
    backends/tex_inspect_opengl.h
    backends/tex_inspect_opengl_loader.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/imgui_tex_inspect")
endif ()

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT ${PROJECT_NAME}Targets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_NAME}Config.cmake)
