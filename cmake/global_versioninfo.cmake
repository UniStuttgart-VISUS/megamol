find_package(Git REQUIRED)

# https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake
execute_process(COMMAND
  ${GIT_EXECUTABLE} describe --match=NeVeRmAtCh --always --abbrev=12 --dirty
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_HASH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file(${CMAKE_SOURCE_DIR}/cmake/MMCoreVersionInfo.cpp.input ${CMAKE_BINARY_DIR}/version/MMCoreVersionInfo.cpp @ONLY)

string(TIMESTAMP YEAR "%Y")
add_definitions(-DMEGAMOL_VERSION_YEAR=${YEAR})
