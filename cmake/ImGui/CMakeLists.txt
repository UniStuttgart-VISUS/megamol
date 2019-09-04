cmake_minimum_required(VERSION 3.0)

project(ImGui CXX)

set(IMGUI_PUBLIC_HDRS
  imgui.h
  imconfig.h)
set(IMGUI_PRIVATE_HDRS
  imgui_internal.h
  imstb_rectpack.h
  imstb_textedit.h
  imstb_truetype.h)
set(IMGUI_SRCS
  imgui.cpp
  imgui_draw.cpp
  imgui_widgets.cpp)

set(IMGUI_GL3_PUBLIC_HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/imgui_impl_opengl3.h
  ${CMAKE_CURRENT_SOURCE_DIR}/misc/cpp/imgui_stdlib.h)
set(IMGUI_GL3_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/imgui_impl_opengl3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(imgui STATIC ${IMGUI_SRCS} ${IMGUI_PUBLIC_HDRS} ${IMGUI_PRIVATE_HDRS})
target_sources(imgui INTERFACE ${IMGUI_GL3_PUBLIC_HDRS} ${IMGUI_GL3_SRCS})

install(TARGETS imgui DESTINATION "lib")
