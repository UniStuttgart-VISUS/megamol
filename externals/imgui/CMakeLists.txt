cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(imgui CXX)

set(IMGUI_PUBLIC_HEADERS
  imconfig.h
  imgui.h)
set(IMGUI_PRIVATE_HEADERS
  imgui_internal.h
  imstb_rectpack.h
  imstb_textedit.h
  imstb_truetype.h
  misc/cpp/imgui_stdlib.h)
set(IMGUI_SOURCES
  imgui.cpp
  imgui_demo.cpp
  imgui_draw.cpp
  imgui_tables.cpp
  imgui_widgets.cpp
  misc/cpp/imgui_stdlib.cpp)
set(IMGUI_BACKEND_HEADERS
  imgui_sw.h
  backends/imgui_impl_generic.h
)
set(IMGUI_BACKEND_SOURCES
  imgui_sw.cpp
  backends/imgui_impl_generic.cpp
)

if (MEGAMOL_USE_OPENGL)
  set(IMGUI_BACKEND_HEADERS ${IMGUI_BACKEND_HEADERS}
    backends/imgui_impl_opengl3.h
    backends/imgui_impl_glfw.h
  )
  set(IMGUI_BACKEND_SOURCES ${IMGUI_BACKEND_SOURCES}
    backends/imgui_impl_opengl3.cpp
    backends/imgui_impl_glfw.cpp
  )
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_PUBLIC_HEADERS} ${IMGUI_PRIVATE_HEADERS} ${IMGUI_BACKEND_SOURCES} ${IMGUI_BACKEND_HEADERS})
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD2)

target_include_directories(imgui PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backends>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc/cpp>
  ${GLAD_INCLUDE_DIR}
  ${GLFW_INCLUDE_DIR})

install(TARGETS imgui DESTINATION "lib")
install(FILES ${IMGUI_PUBLIC_HEADERS} ${IMGUI_PRIVATE_HEADERS} ${IMGUI_BACKEND_HEADERS} DESTINATION "include")
