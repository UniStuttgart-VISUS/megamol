cmake_minimum_required(VERSION 3.0)

project(ogltest)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(vislib REQUIRED HINTS ${vislib_DIR})

add_executable(ogltest ogltest.cpp)
target_link_libraries(ogltest ${GLUT_LIBRARIES} ${vislib_LIBRARIES})
target_include_directories(ogltest PUBLIC ${vislib_INCLUDE_DIRS})

#install(TARGETS ogltest RUNTIME DESTINATION bin)
