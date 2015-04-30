cmake_minimum_required(VERSION 2.8.12)

project(fileiotest)

add_definitions(-Wall -pedantic -ansi -std=c++11)

set(libs)

find_package(vislib REQUIRED)
if (vislib_FOUND)
  list(APPEND libs ${vislib_LIBRARIES})
  include_directories(${vislib_INCLUDE_DIRS})
endif()

add_executable(fileiotest main.cpp)
target_link_libraries(fileiotest ${libs})

install(TARGETS fileiotest RUNTIME DESTINATION bin)
