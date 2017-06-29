# - Find AntTweakBar
# Find the AntTweakBar includes and library
# This module defines
#  AntTweakBar_INCLUDE_DIR, where to find 'AntTweakBar.h', etc.
#  AntTweakBar_LIBRARIES, the libraries needed to use AntTweakBar.
#  AntTweakBar_FOUND, If false, do not try to use AntTweakBar.
# also defined, but not for general use are
#  AntTweakBar_LIBRARY, the files that need to be copied to find the AntTweakBar.
#

# setup of the lib name to search for
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND WIN32)
  set(lib_NAME "AntTweakBar64")
else()
  set(lib_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}AntTweakBar${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

# setup of system search paths
set(lib_search_paths /lib /usr/lib /usr/local/lib $ENV{ANTTWEAKBAR_ROOT}/lib)
set(inc_search_paths /usr/include /usr/local/include $ENV{ANTTWEAKBAR_ROOT}/include)


# setup of additional search hint paths
set(AntTweakBar_DIR "AntTweakBar_DIR-NOTFOUND" CACHE PATH "AntTweakBar directory hint")
set(hint_paths)
get_filename_component(my_dir "${AntTweakBar_DIR}" ABSOLUTE)
list(APPEND hint_paths ${my_dir})


# now perform the actual search
if (NOT AntTweakBar_SEARCH_DEPTH)
	set(AntTweakBar_SEARCH_DEPTH 2)
endif()
foreach(SEARCH_ITERATION RANGE 0 ${AntTweakBar_SEARCH_DEPTH})
	# message(STATUS "Searching in: ${hint_paths}")

	# search
	set(lib_search_hints)
	set(inc_search_hints)
	# construct the hint paths
	foreach(hint_path ${hint_paths})
		list(APPEND lib_search_hints "${hint_path}/lib")
		list(APPEND inc_search_hints "${hint_path}/include")
	endforeach()

	if(WIN32)
		find_file(AntTweakBar_DLL
			NAMES "${lib_NAME}.dll"
			HINTS "${AntTweakBar_DIR}/lib" ${lib_search_hints}
			PATHS ${lib_search_paths}
			)
	endif()

	find_library(AntTweakBar_LIBRARY
		NAMES ${lib_NAME}
		HINTS "${AntTweakBar_DIR}/lib" ${lib_search_hints}
		PATHS ${lib_search_paths}
		)
	find_path(AntTweakBar_INCLUDE_DIR
		NAMES AntTweakBar.h
		HINTS "${AntTweakBar_DIR}/include" ${inc_search_hints}
		PATHS ${inc_search_paths}
		)

	if (AntTweakBar_LIBRARY AND AntTweakBar_INCLUDE_DIR)
		break()
	endif()

	# not found in the current search directories
        # prepare subdirectories for the next iteration
	if (${SEARCH_ITERATION} EQUAL ${AntTweakBar_SEARCH_DEPTH})
		break()
	endif()
	set(next_hint_paths)
	foreach(my_dir in ${hint_paths})
		file(GLOB my_subdirs RELATIVE ${my_dir} "${my_dir}/*")
		foreach(my_subdir ${my_subdirs}) 
			string(SUBSTRING ${my_subdir} 0 1 my_subdir_start)
			get_filename_component(my_full_dir "${my_dir}/${my_subdir}" ABSOLUTE)
			list(FIND hint_paths ${my_full_dir} my_full_dir_found)
			if ((IS_DIRECTORY "${my_full_dir}") AND (NOT ${my_subdir_start} STREQUAL ".") AND (${my_full_dir_found} EQUAL -1))
				#message(STATUS "my_subdir == ${my_full_dir}")
				# add this directory to the hints
				list(APPEND next_hint_paths "${my_full_dir}")
			endif()
		endforeach()
	endforeach()
	set(hint_paths ${next_hint_paths})
endforeach()


# finalizing search
if (AntTweakBar_LIBRARY AND AntTweakBar_INCLUDE_DIR)
	if(WIN32)
		if(AntTweakBar_DLL)
			set(AntTweakBar_LIBRARIES ${AntTweakBar_LIBRARY})
			set(AntTweakBar_FOUND TRUE)
		else()
			set(AntTweakBar_FOUND FALSE)
		endif()
	else()
		set(AntTweakBar_LIBRARIES ${AntTweakBar_LIBRARY})
		set(AntTweakBar_FOUND TRUE)
	endif()
else ()
	set(AntTweakBar_FOUND FALSE)
endif ()


# search result feedback
if (AntTweakBar_FOUND)
	if (NOT AntTweakBar_FIND_QUIETLY)
		message(STATUS "Found AntTweakBar: ${AntTweakBar_LIBRARIES}")
	endif ()
else ()
	if (AntTweakBar_FIND_REQUIRED)
		#message( "library: ${AntTweakBar_LIBRARIES}" )
		#message( "include: ${AntTweakBar_INCLUDE_DIR}" )
		message(FATAL_ERROR "Could not find AntTweakBar library. Please provide at least the AntTweakBar_DIR or set the environmental variable ANTTWEAKBAR_ROOT. ")
	endif ()
endif ()


mark_as_advanced(
	AntTweakBar_LIBRARY
	)

