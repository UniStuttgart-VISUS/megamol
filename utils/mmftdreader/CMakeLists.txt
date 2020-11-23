cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(mmftdreader)
set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Files
set(files
  mmftdreader.cpp)

# Project
add_executable(${PROJECT_NAME} ${files})
if (UNIX)
  target_link_libraries(${PROJECT_NAME} PRIVATE "stdc++fs")
endif ()

# Install
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
