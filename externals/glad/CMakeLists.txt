cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(glad VERSION 1.0.0 LANGUAGES C)

include(GNUInstallDirs)

set(SOURCE_FILES src/glad.c src/glad_glx.c src/glad_wgl.c)
set(HEADER_FILES include/KHR/khrplatform.h include/glad/glad.h include/glad/glad_glx.h include/glad/glad_wgl.h)

if(UNIX)
  list(REMOVE_ITEM SOURCE_FILES "src/glad_wgl.c")
  list(REMOVE_ITEM HEADER_FILES "include/glad/glad_wgl.h")
elseif(WIN32)
  list(REMOVE_ITEM SOURCE_FILES "src/glad_glx.c")
  list(REMOVE_ITEM HEADER_FILES "include/glad/glad_glx.h")
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_compile_definitions(${PROJECT_NAME} 
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC GLAD_GLAPI_EXPORT)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_include_directories(${PROJECT_NAME} 
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1)

install(TARGETS ${PROJECT_NAME} EXPORT gladConfig
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/)

install(EXPORT gladConfig DESTINATION share/glad/cmake)

export(TARGETS ${PROJECT_NAME} FILE gladConfig.cmake)
