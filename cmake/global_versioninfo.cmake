
# https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake
execute_process(COMMAND
  "git" describe --match=NeVeRmAtCh --always --abbrev=12 --dirty
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_HASH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)


# VISLIB VERSION
set(VISLIB_MAJOR 2)
set(VISLIB_MINOR 2)
add_definitions(-DVISLIB_VERSION_MAJOR=${VISLIB_MAJOR})
add_definitions(-DVISLIB_VERSION_MINOR=${VISLIB_MINOR})
add_definitions(-DVISLIB_VERSION_REVISION="${GIT_HASH}")


# MEGAMOL CORE VERSION
set(MEGAMOL_MAJOR 1)
set(MEGAMOL_MINOR 2)
set(MEGAMOL_PATCH 0)
set(MEGAMOL_VERSION ${MEGAMOL_MAJOR}.${MEGAMOL_MINOR}.${MEGAMOL_PATCH})

add_definitions(-DMEGAMOL_VERSION_MAJOR=${MEGAMOL_MAJOR})
add_definitions(-DMEGAMOL_VERSION_MINOR=${MEGAMOL_MINOR})
add_definitions(-DMEGAMOL_VERSION_PATCH=${MEGAMOL_PATCH})
add_definitions(-DMEGAMOL_CORE_COMP_REV="${GIT_HASH}")

