cmake_minimum_required(VERSION 3.8)

project(obj-io)

file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")

add_library(obj-io INTERFACE ${header_files})
target_include_directories(obj-io INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)
install(FILES ${header_files} DESTINATION "${CMAKE_INSTALL_PREFIX}/include")