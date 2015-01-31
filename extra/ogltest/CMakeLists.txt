cmake_minimum_required(VERSION 2.8)

project(ogltest)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(vislib REQUIRED)
include_directories(${vislib_INCLUDE_DIRS})

add_executable(ogltest ogltest.cpp)
target_link_libraries(ogltest ${GLUT_LIBRARIES} ${vislib_LIBRARIES})


#install(TARGETS ogltest RUNTIME DESTINATION bin)
