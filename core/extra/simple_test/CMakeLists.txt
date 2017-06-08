cmake_minimum_required(VERSION 2.6)
project(simple_test)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(MegaMolCore REQUIRED)
set(LIBS ${MegaMolCore_LIBRARIES})
include_directories(${MegaMolCore_INCLUDE_DIRS})

add_definitions(-std=c++11)

add_executable(simple_test main.cpp)
target_link_libraries(simple_test ${LIBS})

#install(TARGETS simple_test RUNTIME DESTINATION bin)
