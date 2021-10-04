find_package(Git REQUIRED)

# https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake
execute_process(COMMAND
  ${GIT_EXECUTABLE} describe --match=NeVeRmAtCh --always --abbrev=12 --dirty
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_HASH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND
  ${GIT_EXECUTABLE} status -sb
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_STATUS_SB
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX MATCH "^## [a-zA-Z/_]+\.\.\.([a-zA-Z/_]+)" _ ${GIT_STATUS_SB})
set(GIT_STATUS_REMOTE ${CMAKE_MATCH_1})
string(REGEX MATCH "([a-zA-Z]+)/[a-zA-Z]+" _ ${GIT_STATUS_REMOTE})
unset(GIT_STATUS_SB)

execute_process(COMMAND
  ${GIT_EXECUTABLE} remote get-url ${CMAKE_MATCH_1}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_REMOTE_URL
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file(${CMAKE_SOURCE_DIR}/cmake/MMCoreVersionInfo.cpp.input ${CMAKE_BINARY_DIR}/version/MMCoreVersionInfo.cpp @ONLY)

string(TIMESTAMP YEAR "%Y")
add_definitions(-DMEGAMOL_VERSION_YEAR=${YEAR})
