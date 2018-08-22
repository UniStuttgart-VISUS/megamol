include(CMakeParseArguments)
include(ExternalProject)

function(argument_default VARIABLE)
  if(args_${VARIABLE})
    set(${VARIABLE} ${args_${VARIABLE}} PARENT_SCOPE)
  else()
    set(${VARIABLE} ${ARGN} PARENT_SCOPE)
  endif()
endfunction(argument_default)

function(add_external_library TARGET)
  set(ARG_OPTIONS STATIC SHARED INTERFACE)
  set(ARG_SINGLE LIBRARY_DEBUG LIBRARY_RELEASE INCLUDE_DIR COMPILE_DEFINITIONS)
  set(ARG_SINGLE_EXT GIT_TAG)
  cmake_parse_arguments(args "${ARG_OPTIONS}" "${ARG_SINGLE};${ARG_SINGLE_EXT}" "" ${ARGN})

  # Filter arguments for ExternalProject_Add.
  set(ARGN_EXT ${ARGN})
  list(REMOVE_ITEM ARGN_EXT ${ARG_OPTIONS})
  foreach(arg ${ARG_SINGLE})
    list(FIND ARGN_EXT ${arg} index)
    if(NOT index EQUAL -1)
      # Remove key and succeding value.
      list(REMOVE_AT ARGN_EXT ${index})
      list(REMOVE_AT ARGN_EXT ${index})
    endif()
  endforeach()

  # Add external project.
  set(INSTALL_PREFIX_EXT="${CMAKE_BINARY_DIR}/external")
  set(TARGET_EXT "${TARGET}_ext")
  ExternalProject_Add(${TARGET_EXT} ${ARGN_EXT}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX_EXT})

  if(args_GIT_TAG)
    message(STATUS "${TARGET} tag: ${args_GIT_TAG}")
  endif()

  # Guess target properties, unless set.
  ExternalProject_Get_Property(${TARGET_EXT} INSTALL_DIR)
  argument_default(LIBRARY_DEBUG ${CMAKE_STATIC_LIBRARY_PREFIX}${TARGET}${CMAKE_STATIC_LIBRARY_SUFFIX})
  argument_default(LIBRARY_RELEASE ${LIBRARY_DEBUG})
  argument_default(INCLUDE_DIR include)
  argument_default(COMPILE_DEFINITIONS "")

  # Set linkage.
  if(args_STATIC)
    set(LINKAGE STATIC)
  elseif(args_SHARED)
    set(LINKAGE SHARED)
  elseif(args_INTERFACE)
    set(LINKAGE INTERFACE)
  endif()

  # Add import library.
  add_library(${TARGET} ${LINKAGE} IMPORTED)
  add_dependencies(${TARGET} ${TARGET_EXT})
  set_target_properties(${TARGET} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${INSTALL_DIR}/${INCLUDE_DIR}"
    INTERFACE_COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}")
  if(LINKAGE STREQUAL "STATIC" OR LINKAGE STREQUAL "SHARED")
    set_target_properties(${TARGET} PROPERTIES
      IMPORTED_CONFIGURATIONS "Debug;Release"
      IMPORTED_LOCATION_DEBUG "${INSTALL_DIR}/${LIBRARY_DEBUG}"
      IMPORTED_LOCATION_RELEASE "${INSTALL_DIR}/${LIBRARY_RELEASE}")
      #IMPORTED_NO_SONAME_DEBUG
	  #IMPORTED_NO_SONAME_RELEASE
  endif()
endfunction(add_external_library)
