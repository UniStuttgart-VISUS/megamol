cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(glad
  VERSION 2.0.0
  LANGUAGES C)

set(SOURCE_FILES
  src/gl.c
  src/vulkan.c)
set(HEADER_FILES
  include/glad/gl.h
  include/glad/vulkan.h
  include/KHR/khrplatform.h
  include/vk_platform.h)

if (UNIX)
  list(APPEND SOURCE_FILES "src/glx.c")
  list(APPEND HEADER_FILES "include/glad/glx.h")
elseif (WIN32)
  list(APPEND SOURCE_FILES "src/wgl.c")
  list(APPEND HEADER_FILES "include/glad/wgl.h")
endif ()

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${HEADER_FILES})

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Install
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME} EXPORT gladConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT gladConfig
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/glad)

export(TARGETS ${PROJECT_NAME} FILE gladConfig.cmake)
