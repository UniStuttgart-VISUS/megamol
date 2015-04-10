# - Find visglut
# Find the modified visglut includes and library
# This module defines
#  visglut_INCLUDE_DIR, where to find 'visglut.h', etc.
#  visglut_LIBRARIES, the libraries needed to use visglut.
#  visglut_FOUND, If false, do not try to use visglut.
# also defined, but not for general use are
#  visglut_NEED_TO_COPY, the files that need to be copied to find the visglut
#

# setup of the lib name to search for
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(BITS 64)
else()
	set(BITS 32)
endif()
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	set(lib_POSTFIX "d")
else()
	set(lib_POSTFIX "")
endif()
set(lib_NAME "visglut${BITS}${lib_POSTFIX}")


# setup of system search paths
set(lib_search_paths /lib /usr/lib /usr/local/lib)
set(inc_search_paths /usr/include /usr/local/include)


# setup of additional search hint paths
set(hint_paths ${CMAKE_CURRENT_SOURCE_DIR})
if (visglut_DIR)
	list(APPEND hint_paths ${visglut_DIR})
endif()
# traverse file system up to the second-highest level
get_filename_component(my_dir "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
while (${my_dir} STRGREATER "/")
	#message(STATUS "my_dir == ${my_dir}")

	# check on directory level below
	file(GLOB my_subdirs RELATIVE ${my_dir} "${my_dir}/*")
	foreach(my_subdir ${my_subdirs}) 

		# only check non-hidden directories
		string(SUBSTRING ${my_subdir} 0 1 my_subdir_start)
		if ((IS_DIRECTORY "${my_dir}/${my_subdir}") AND (NOT ${my_subdir_start} STREQUAL "."))
			#message(STATUS "my_subdir == ${my_subdir}")

			# add this directory to the hints
			list(APPEND hint_paths "${my_dir}/${my_subdir}")
		endif()
	endforeach()
	get_filename_component(my_dir "${my_dir}/.." ABSOLUTE)
endwhile()
# construct the hint paths
foreach(hint_path ${hint_paths})
	list(APPEND lib_search_hints "${hint_path}/lib")
	list(APPEND inc_search_hints "${hint_path}/include")
endforeach()


# perform the search
#message(STATUS "Searching visglut lib in: ${lib_search_hints};${lib_search_paths}")
#message(STATUS "Searching visglut includes in: ${inc_search_hints};${inc_search_paths}")
find_library(visglut_NEED_TO_COPY
	NAMES ${lib_NAME}
	HINTS "${vislut_DIR}/lib" ${lib_search_hints}
	PATHS ${lib_search_paths}
	)
find_path(visglut_INCLUDE_DIR
	NAMES visglut.h GL/glut.h
	HINTS "${visglut_NEED_TO_COPY}/../include" ${inc_search_hints}
	PATHS ${inc_search_paths}
	)

# finalizing search
if (visglut_NEED_TO_COPY AND visglut_INCLUDE_DIR)
	set(visglut_LIBRARIES ${visglut_NEED_TO_COPY})
	set(visglut_FOUND TRUE)
else ()
	set(visglut_FOUND FALSE)
endif ()

# search result feedback
if (visglut_FOUND)
	if (NOT visglut_FIND_QUIETLY)
		message(STATUS "Found visglut: ${visglut_LIBRARIES}")
	endif ()
else ()
	if (visglut_FIND_REQUIRED)
		#message( "library: ${visglut_LIBRARIES}" )
		#message( "include: ${visglut_INCLUDE_DIR}" )
		message(FATAL_ERROR "Could not find visglut library")
	endif ()
endif ()

mark_as_advanced(
	visglut_NEED_TO_COPY
	visglut_INCLUDE_DIR
	)

