project(glad)

include_directories("include" "src")
set(LIBS ${OPENGL_LIBRARIES})
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c")

if(UNIX)
  list(REMOVE_ITEM source_files "src/glad_wgl.c")
  list(REMOVE_ITEM header_files "include/glad/glad_wgl.h")
elseif(WIN32)
  list(REMOVE_ITEM source_files "src/glad_glx.c")
  list(REMOVE_ITEM header_files "include/glad/glad_glx.h")
endif()

add_library(glad SHARED ${header_files} ${source_files})
target_link_libraries(glad PRIVATE ${LIBS})
target_include_directories(glad PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE CXX)

target_compile_definitions(glad PRIVATE GLAD_GLAPI_EXPORT_BUILD)
target_compile_definitions(glad PUBLIC GLAD_GLAPI_EXPORT)

# Installation rules for generated files
if(WIN32)
  install(TARGETS glad RUNTIME DESTINATION "bin")
  install(TARGETS glad ARCHIVE DESTINATION "lib")
else()
  install(TARGETS glad DESTINATION "lib" EXPORT glad-target)
endif()

set_target_properties(glad PROPERTIES FOLDER external)