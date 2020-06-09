project(glad_)

file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c")

if(UNIX)
  list(REMOVE_ITEM source_files "src/glad_wgl.c")
  list(REMOVE_ITEM header_files "include/glad/glad_wgl.h")
elseif(WIN32)
  list(REMOVE_ITEM source_files "src/glad_glx.c")
  list(REMOVE_ITEM header_files "include/glad/glad_glx.h")
endif()

add_library(glad_ SHARED ${header_files} ${source_files})
set_target_properties(glad_ PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(glad_ 
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC GLAD_GLAPI_EXPORT)
target_include_directories(glad_ PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  "include" "src")
target_link_libraries(glad_ PRIVATE ${OPENGL_LIBRARIES})

# Installation rules for generated files
if(WIN32)
  install(TARGETS glad_ RUNTIME DESTINATION "bin")
  install(TARGETS glad_ ARCHIVE DESTINATION "lib")
else()
  install(TARGETS glad_ DESTINATION "lib" EXPORT glad-target)
endif()
