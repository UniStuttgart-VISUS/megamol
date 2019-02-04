#
# MegaMol Test Suite
# Copyright 2019, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#

project(mm_tests)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Check option dependencies: First list item is always the current build option,
# afterwards list the dependencies.
set(DEP_LIST "${DEP_LIST};BUILD_CORE" CACHE INTERNAL "")

include(External)
    bootstrap_external_package(googletest_ext GTest
      CONFIG "lib/cmake/GTest/GTestConfig.cmake"
      GIT_REPOSITORY    https://github.com/google/googletest.git
      GIT_TAG           master
    )
find_package(GTest REQUIRED CONFIG)

file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")

add_executable(${PROJECT_NAME} ${header_files} ${source_files})
set_target_properties(${PROJECT_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "src" "include")
target_link_libraries(${PROJECT_NAME} 
    PRIVATE GTest::gtest core)

# Source groups in Visual Studio
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
foreach(FILE_NAME ${source_files})
  get_filename_component(FILE_PATH ${FILE_NAME} PATH)
  string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
  string(REGEX REPLACE "^src" "Source Files" GROUP_NAME ${GROUP_NAME})
  source_group(${GROUP_NAME} FILES ${FILE_NAME})
endforeach()
foreach(FILE_NAME ${header_files})
  get_filename_component(FILE_PATH ${FILE_NAME} PATH)
  string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
  string(REGEX REPLACE "^src" "Header Files" GROUP_NAME ${GROUP_NAME})
  source_group(${GROUP_NAME} FILES ${FILE_NAME})
endforeach()
foreach(FILE_NAME ${public_header_files})
  get_filename_component(FILE_PATH ${FILE_NAME} PATH)
  string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
  string(REGEX REPLACE "^include\\\\mmcore" "Public Header Files" GROUP_NAME ${GROUP_NAME})
  source_group(${GROUP_NAME} FILES ${FILE_NAME})
endforeach()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
