project(datraw)

# Collect source files
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c")

# Target definition
add_library(${PROJECT_NAME} STATIC ${header_files} ${source_files})
target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_VSNPRINTF)
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  "include" "src")
target_link_libraries(${PROJECT_NAME} PRIVATE core)

install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
