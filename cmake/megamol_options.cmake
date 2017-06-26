# compiler options
if(WIN32)
  # avoid problematic min/max defines of windows.h
  add_definitions(-DNOMINMAX)
  # options
  add_definitions(-W3 -pedantic -ansi -fPIC)
elseif(UNIX)
  # processor word size detection
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BITS 64)
  else()
    set(BITS 32)
  endif()
  add_definitions(-Wall -pedantic -ansi -fPIC -DUNIX -D_GNU_SOURCE -D_LIN${BITS})
endif()
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
	message(STATUS "Version < 4.7")
	add_definitions(-std=c++0x)
else()
	add_definitions(-std=c++11)
endif()

# Set CXX flags for debug mode
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -D_DEBUG -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -D_NDEBUG -O3 -g0")


set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(MEGAMOL_DEBUG_BUILD ON)
  set(MEGAMOL_RELEASE_BUILD OFF)
else()# Release
  set(MEGAMOL_DEBUG_BUILD OFF)
  set(MEGAMOL_RELEASE_BUILD ON)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})


option(MEGAMOL_INSTALL_DEPENDENCIES "MegaMol dependencies in install" ON)
mark_as_advanced(MEGAMOL_INSTALL_DEPENDENCIES)

# Global Packages
set(CMAKE_THREAD_PREFER_PTHREAD)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
