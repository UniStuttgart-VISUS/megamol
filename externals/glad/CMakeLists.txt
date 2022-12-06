cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(glad
  VERSION 2.0.0
  LANGUAGES C)

set(source_files
  src/gl.c
  src/vulkan.c)
set(header_files
  include/glad/gl.h
  include/glad/vulkan.h
  include/KHR/khrplatform.h
  include/vk_platform.h)

if (UNIX)
  list(APPEND source_files "src/glx.c")
  list(APPEND header_files "include/glad/glx.h")
elseif (WIN32)
  list(APPEND source_files "src/wgl.c")
  list(APPEND header_files "include/glad/wgl.h")
endif ()

add_library(${PROJECT_NAME} STATIC ${source_files} ${header_files})

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_DL_LIBS})
