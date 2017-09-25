project(lua C)

set(public_header_files lua.h lauxlib.h lualib.h)
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c")
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")


add_library(${PROJECT_NAME} SHARED ${source_files} ${header_files})
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_compile_definitions(${PROJECT_NAME} PRIVATE LUA_LIB)

if(WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE LUA_BUILD_AS_DLL)
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
else()
  install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER external)