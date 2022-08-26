cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(imguitexinspect CXX)

set(IMGUITEXINSPECT_PUBLIC_HDRS
  imgui_tex_inspect_internal.h
  imgui_tex_inspect.h
  backends/tex_inspect_opengl.h)
set(IMGUITEXINSPECT_PRIVATE_HDRS)
set(IMGUITEXINSPECT_SRCS
  imgui_tex_inspect.cpp
  backends/tex_inspect_opengl.cpp)

#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(imguitexinspect STATIC ${IMGUITEXINSPECT_SRCS} ${IMGUITEXINSPECT_PUBLIC_HDRS} ${IMGUITEXINSPECT_PRIVATE_HDRS})
target_compile_definitions(imguitexinspect PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD2)
target_link_libraries(imguitexinspect PRIVATE ${IMGUI_LIBRARY} ${GLAD_LIBRARY})
target_include_directories(imguitexinspect PRIVATE ${IMGUI_INCLUDE_DIR} ${GLAD_INCLUDE_DIR} ${SOURCE_INCLUDE_DIR})

install(TARGETS imguitexinspect DESTINATION "lib")
install(FILES ${IMGUITEXINSPECT_PUBLIC_HDRS} DESTINATION "include")
