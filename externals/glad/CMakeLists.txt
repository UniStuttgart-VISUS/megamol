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

# set_target_properties(glad PROPERTIES LINKER_LANGUAGE C)
# set_target_properties(glad PROPERTIES PUBLIC_HEADER ${header_files})
# 
# target_compile_definitions(glad 
#   PUBLIC GLAD_GLAPI_EXPORT_BUILD
#   PUBLIC GLAD_GLAPI_EXPORT)
# 
# #set_target_properties(glad PROPERTIES INTERFACE_COMPILE_DEFINITIONS "GLAD_GLAPI_EXPORT_BUILD" "GLAD_GLAPI_EXPORT")
# 
# target_include_directories(glad 
#   PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src"
#   $<INSTALL_INTERFACE:include>)
# 
# target_link_libraries(glad PRIVATE ${OPENGL_LIBRARIES})
# 
# include(GNUInstallDirs)
# 
# install(TARGETS glad
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# 
# # Installation rules for generated files
# if(WIN32)
#   install(TARGETS glad RUNTIME DESTINATION "bin")
#   install(TARGETS glad ARCHIVE DESTINATION "lib")
# else()
#   install(TARGETS glad DESTINATION "lib" EXPORT glad-target)
# endif()
