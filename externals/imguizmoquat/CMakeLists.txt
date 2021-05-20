cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(imguizmoquat CXX)

set(IMGUIZMOQUAT_PUBLIC_HEADERS
  imGuIZMO.quat/imGuIZMOquat.h)
set(IMGUIZMOQUAT_PRIVATE_HEADERS
  imGuIZMO.quat/vgConfig.h
  imGuIZMO.quat/vGizmo.h
  imGuIZMO.quat/vGizmoMath.h
  imGuIZMO.quat/vgMath.h)
set(IMGUIZMOQUAT_SOURCES
  imGuIZMO.quat/imGuIZMOquat.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(imguizmoquat STATIC ${IMGUIZMOQUAT_SOURCES} ${IMGUIZMOQUAT_PUBLIC_HEADERS} ${IMGUIZMOQUAT_PRIVATE_HEADERS})
target_compile_definitions(imguizmoquat PRIVATE IMGUIZMO_IMGUI_FOLDER=./)
target_include_directories(imguizmoquat PRIVATE imGuIZMO.quat ${IMGUI_INCLUDE_DIR})

install(TARGETS imguizmoquat DESTINATION "lib")
install(FILES ${IMGUIZMOQUAT_PUBLIC_HEADERS} ${IMGUIZMOQUAT_PRIVATE_HEADERS} DESTINATION "include")
