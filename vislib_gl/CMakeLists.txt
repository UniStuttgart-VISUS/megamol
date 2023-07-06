project(vislib_gl)

# Build vislib
file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

# Target definition
add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files})

find_package(glad REQUIRED)

target_link_libraries(${PROJECT_NAME}
  PUBLIC glad::glad ${OPENGL_LIBRARIES} vislib)

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src")

if (UNIX)
  find_package(X11 REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_LIBRARIES})
endif ()

# Grouping in Visual Studio
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
source_group("Public Header Files" FILES ${public_header_files})
source_group("Header Files" FILES ${header_files})
source_group("Source Files" FILES ${source_files})
