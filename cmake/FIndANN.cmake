# - Find ANN
# Find the ANN includes and library
# This module defines
#  ANN_INCLUDE_DIR, where to find 'ANN.h', etc.
#  ANN_LIBRARIES, the libraries needed to use ANN.
#  ANN_FOUND, If false, do not try to use ANN.
#

set(ANN_INSTALL_DIR $ENV{ANN_INSTALL_DIR} CACHE PATH "Path to ANN installed location.")

# setup of hint paths
set(lib_search_hints )
set(inc_search_hints )
if (ANN_INSTALL_DIR)
	set(lib_search_hints "${ANN_INSTALL_DIR}" "${ANN_INSTALL_DIR}/lib")
	set(inc_search_hints "${ANN_INSTALL_DIR}" "${ANN_INSTALL_DIR}/include")
endif()

if (WIN32)
  set(dll_search_hints "${ANN_INSTALL_DIR}" "${ANN_INSTALL_DIR}/bin")
else()
  set(dll_search_hints "")
endif()


# setup of system search paths
set(lib_search_paths /lib /usr/lib /usr/local/lib)
set(inc_search_paths /usr/include /usr/local/include)

if (WIN32)
  set(CMAKE_FIND_LIBRARY_PREFIXES "")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
else()
  set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so")
endif()

# perform the search
find_library(ANN_LIBRARIES
  ANN
  HINTS    
	${lib_search_hints})
if (WIN32)
find_library(ANN_DLL
  ANN
  HINTS    
    ${dll_search_hints})
endif()
# if (WIN32)
#   find_library(ANN_LIBRARIES
#     NAMES ANN.lib
#     HINTS ${lib_search_hints})
#   find_file(ANN_DLL
#     NAMES ANN.dll
#     HINTS ${dll_search_hints})
# else()
#   find_library(ANN_LIBRARIES
#     NAMES libANN.so
#     HINTS ${lib_search_hints}
#     PATHS ${lib_search_paths})
# endif()
find_path(ANN_INCLUDE_DIR
	NAMES ANN/ANN.h
	HINTS ${inc_search_hints}
	PATHS ${inc_search_paths}
	)

# finalizing search
if (ANN_LIBRARIES AND ANN_INCLUDE_DIR)
	set(ANN_FOUND TRUE)
else ()
	set(ANN_FOUND FALSE)
endif ()

# search result feedback
if (ANN_FOUND)
	if (NOT ANN_FIND_QUIETLY)
		message(STATUS "Found ANN: ${ANN_LIBRARIES}")
	endif ()
else ()
	if (ANN_FIND_REQUIRED)
		#message( "library: ${ANN_LIBRARIES}" )
		#message( "include: ${ANN_INCLUDE_DIR}" )
		message(FATAL_ERROR "Could not find ANN library")
	endif ()
endif ()