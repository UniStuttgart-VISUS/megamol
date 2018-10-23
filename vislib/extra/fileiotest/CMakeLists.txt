cmake_minimum_required(VERSION 3.0)

project(fileiotest)

add_executable(fileiotest main.cpp)

find_package(vislib REQUIRED)
if (vislib_FOUND)
  target_link_libraries(fileiotest ${vislib_LIBRARIES})
  target_include_directories(fileiotest PUBLIC ${vislib_INCLUDE_DIRS})
endif()

install(TARGETS fileiotest RUNTIME DESTINATION bin)
