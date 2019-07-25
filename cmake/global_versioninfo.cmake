find_package(Git REQUIRED)

# https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake
execute_process(COMMAND
  ${GIT_EXECUTABLE} describe --match=NeVeRmAtCh --always --abbrev=12 --dirty
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_HASH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)


# VISLIB VERSION
set(VISLIB_MAJOR 2)
set(VISLIB_MINOR 2)
configure_file(${CMAKE_SOURCE_DIR}/cmake/VISlibVersionInfo.cpp.input ${CMAKE_BINARY_DIR}/version/VISlibVersionInfo.cpp @ONLY)


# MEGAMOL CORE VERSION
set(MEGAMOL_MAJOR 1)
set(MEGAMOL_MINOR 3)
set(MEGAMOL_PATCH 0)
set(MEGAMOL_VERSION ${MEGAMOL_MAJOR}.${MEGAMOL_MINOR}.${MEGAMOL_PATCH})

configure_file(${CMAKE_SOURCE_DIR}/cmake/MMCoreVersionInfo.cpp.input ${CMAKE_BINARY_DIR}/version/MMCoreVersionInfo.cpp @ONLY)

string(TIMESTAMP YEAR "%Y")
add_definitions(-DMEGAMOL_VERSION_YEAR=${YEAR})
