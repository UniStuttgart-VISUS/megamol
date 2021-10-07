# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Directory of the current script
get_filename_component(INFO_SRC_DIR ${CMAKE_SCRIPT_MODE_FILE} DIRECTORY)

# MegaMol project root
get_filename_component(PROJECT_DIR ${INFO_SRC_DIR} DIRECTORY)
get_filename_component(PROJECT_DIR ${PROJECT_DIR} DIRECTORY)

# Find git
find_package(Git REQUIRED)

# Hash
execute_process(COMMAND
  ${GIT_EXECUTABLE} describe --match=NeVeRmAtCh --always --abbrev=16 --dirty
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_HASH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# Branch name
execute_process(COMMAND
  ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_BRANCH_NAME
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# Remote branch name
execute_process(COMMAND
  ${GIT_EXECUTABLE} rev-parse --abbrev-ref --symbolic-full-name @{u}
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_BRANCH_NAME_FULL
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get Origin name
if (GIT_BRANCH_NAME_FULL STREQUAL "")
  set(GIT_ORIGIN_NAME "")
else ()
  string(REPLACE "/" ";" GIT_ORIGIN_NAME "${GIT_BRANCH_NAME_FULL}")
  list(GET GIT_ORIGIN_NAME 0 GIT_ORIGIN_NAME)
endif ()

# Origin URL
execute_process(COMMAND
  ${GIT_EXECUTABLE} remote get-url "${GIT_ORIGIN_NAME}"
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_REMOTE_URL
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# Git diff
execute_process(COMMAND
  ${GIT_EXECUTABLE} diff HEAD
  WORKING_DIRECTORY "${PROJECT_DIR}"
  OUTPUT_VARIABLE GIT_DIFF
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# Time
string(TIMESTAMP BUILD_TIMESTAMP "%s" UTC)
string(TIMESTAMP BUILD_TIME "" UTC)

# License
file(READ ${PROJECT_DIR}/LICENSE MEGAMOL_LICENSE)

# Cache
file(READ ${CMAKE_BINARY_DIR}/CMakeCache.txt MM_CMAKE_CACHE)

# Write to sourcefile
configure_file(${INFO_SRC_DIR}/megamol_build_info_buildtime.h.in ${CMAKE_BINARY_DIR}/megamol_build_info/megamol_build_info_buildtime.h @ONLY)
