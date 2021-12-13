# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Functions
function(write_file_if_changed filename content)
  if (EXISTS "${filename}")
    file(READ ${filename} content_old)
    if ("${content}" STREQUAL "${content_old}")
      # File exists already with same content. Do not write, to not trigger unnecessary rebuild.
      return()
    endif ()
  endif ()
  file(WRITE ${filename} "${content}")
endfunction()

# Directory of the current script
get_filename_component(INFO_SRC_DIR ${CMAKE_SCRIPT_MODE_FILE} DIRECTORY)

# MegaMol project root
get_filename_component(PROJECT_DIR ${INFO_SRC_DIR} DIRECTORY)
get_filename_component(PROJECT_DIR ${PROJECT_DIR} DIRECTORY)

set(INFO_RESOURCES_DIR "${CMAKE_BINARY_DIR}/megamol_build_info")

# Find git
find_package(Git REQUIRED)

# Hash
execute_process(COMMAND
  ${GIT_EXECUTABLE} describe --match=NeVeRmAtCh --always --abbrev=16 --dirty
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_HASH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_GIT_HASH "${GIT_HASH}")

# Branch name
execute_process(COMMAND
  ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_BRANCH_NAME
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_GIT_BRANCH_NAME "${GIT_BRANCH_NAME}")

# Full branch name
execute_process(COMMAND
  ${GIT_EXECUTABLE} rev-parse --abbrev-ref --symbolic-full-name @{u}
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_BRANCH_NAME_FULL
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_GIT_BRANCH_NAME_FULL "${GIT_BRANCH_NAME_FULL}")

# Get remote name
if (GIT_BRANCH_NAME_FULL STREQUAL "")
  set(GIT_REMOTE_NAME "")
else ()
  string(REPLACE "/" ";" GIT_REMOTE_NAME "${GIT_BRANCH_NAME_FULL}")
  list(GET GIT_REMOTE_NAME 0 GIT_REMOTE_NAME)
endif ()
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_GIT_REMOTE_NAME "${GIT_REMOTE_NAME}")

# Origin URL
execute_process(COMMAND
  ${GIT_EXECUTABLE} remote get-url "${GIT_REMOTE_NAME}"
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_REMOTE_URL
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_GIT_REMOTE_URL "${GIT_REMOTE_URL}")

# Git diff / is dirty
execute_process(COMMAND
  ${GIT_EXECUTABLE} diff --exit-code HEAD
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_DIFF
  RESULTS_VARIABLE GIT_IS_DIRTY
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_GIT_DIFF "${GIT_DIFF}")
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_GIT_IS_DIRTY "${GIT_IS_DIRTY}")

# Time
string(TIMESTAMP BUILD_TIMESTAMP "%s" UTC)
string(TIMESTAMP BUILD_TIME "" UTC)
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_BUILD_TIMESTAMP "${BUILD_TIMESTAMP}")
write_file_if_changed(${INFO_RESOURCES_DIR}/MEGAMOL_BUILD_TIME "${BUILD_TIME}")

# License
configure_file(${PROJECT_DIR}/LICENSE ${INFO_RESOURCES_DIR}/MEGAMOL_LICENSE COPYONLY)

# Cache
configure_file(${CMAKE_BINARY_DIR}/CMakeCache.txt ${INFO_RESOURCES_DIR}/MEGAMOL_CMAKE_CACHE COPYONLY)
