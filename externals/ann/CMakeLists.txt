cmake_minimum_required(VERSION 3.10)

project(ann)


file(GLOB source_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB header_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE public_header_files "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

message(STATUS "FOUND SOURCE FILES ${source_files}" )
message(STATUS "FOUND PUBLIC_HEADER FILES ${public_header_files}" )


include_directories("include" "src")

add_library(${PROJECT_NAME} SHARED ${source_files} ${header_files})

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(${PROJECT_NAME} PUBLIC DLL_EXPORTS)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_header_files}")

install (TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install (TARGETS ${PROJECT_NAME} PUBLIC_HEADER DESTINATION include/ANN)

