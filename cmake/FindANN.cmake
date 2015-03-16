# - Find ANN
# Find the ANN includes and library
# This module defines
#  ANN_INCLUDE_DIR, where to find 'visglut.h', etc.
#  ANN_LIBRARIES, the libraries needed to use visglut.
#  ANN_FOUND, If false, do not try to use visglut.
#


# setup of system search paths
set(lib_search_paths /lib /usr/lib /usr/local/lib)
set(inc_search_paths /usr/include /usr/local/include)


# perform the search
find_library(ANN_LIBRARIES
	NAMES libann.so
	#HINTS ${lib_search_hints}
	PATHS ${lib_search_paths}
	)
find_path(ANN_INCLUDE_DIR
	NAMES ANN/ANN.h
	#HINTS ${inc_search_hints}
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

