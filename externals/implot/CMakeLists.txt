cmake_minimum_required(VERSION 3.0)

project(implot CXX)

set(IMPLOT_PUBLIC_HDRS
  implot.h)
set(IMPLOT_PRIVATE_HDRS
  implot_internal.h)
set(IMPLOT_SRCS
  implot.cpp
  implot_items.cpp)

add_library(implot STATIC ${IMPLOT_SRCS} ${IMPLOT_PUBLIC_HDRS} ${IMPLOT_PRIVATE_HDRS})

target_compile_definitions(implot PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(implot PRIVATE ${IMGUI_LIBRARY})
target_include_directories(implot PRIVATE ${IMGUI_INCLUDE_DIR})

install(TARGETS implot DESTINATION "lib")
install(FILES ${IMPLOT_PUBLIC_HDRS} DESTINATION "include")
