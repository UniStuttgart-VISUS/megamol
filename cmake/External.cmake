include(CMakeParseArguments)
include(ExternalProject)

function(argument_default VARIABLE)
  if(args_${VARIABLE})
    set(${VARIABLE} "${args_${VARIABLE}}" PARENT_SCOPE)
  else()
    set(${VARIABLE} "${ARGN}" PARENT_SCOPE)
  endif()
endfunction(argument_default)

function(add_external_library TARGET)
  set(ARG_OPTIONS STATIC SHARED INTERFACE)
  set(ARG_SINGLE LIBRARY_DEBUG LIBRARY_RELEASE IMPORT_LIBRARY_DEBUG IMPORT_LIBRARY_RELEASE INCLUDE_DIR COMPILE_DEFINITIONS)
  set(ARG_SINGLE_EXT GIT_TAG)
  cmake_parse_arguments(args "${ARG_OPTIONS}" "${ARG_SINGLE};${ARG_SINGLE_EXT}" "" ${ARGN})

  # Map arguments for ExternalProject_Add.
  set(ARGN_EXT "${ARGN}")
  list(REMOVE_ITEM ARGN_EXT ${ARG_OPTIONS})
  foreach(arg ${ARG_SINGLE})
    list(FIND ARGN_EXT ${arg} index)
    if(NOT index EQUAL -1)
      # Remove key and succeding value.
      list(REMOVE_AT ARGN_EXT ${index})
      list(REMOVE_AT ARGN_EXT ${index})
    endif()
  endforeach()
  list(APPEND ARGN_EXT CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>)

  # Add external project.
  set(TARGET_EXT "${TARGET}_ext")
  ExternalProject_Add(${TARGET_EXT} "${ARGN_EXT}")

  if(args_GIT_TAG)
    message(STATUS "${TARGET} tag: ${args_GIT_TAG}")
  endif()

  # Guess target properties, unless set.
  ExternalProject_Get_Property(${TARGET_EXT} INSTALL_DIR)
  argument_default(LIBRARY_DEBUG ${TARGET})
  argument_default(LIBRARY_RELEASE ${LIBRARY_DEBUG})
  argument_default(IMPORT_LIBRARY_DEBUG ${TARGET})
  argument_default(IMPORT_LIBRARY_RELEASE ${IMPORT_LIBRARY_DEBUG})
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

  # Create include directory as required by INTERFACE_INCLUDE_DIRECTORIES.
  file(MAKE_DIRECTORY "${INSTALL_DIR}/${INCLUDE_DIR}")

  # Add as imported library.
  add_library(${TARGET} ${LINKAGE} IMPORTED GLOBAL)
  add_dependencies(${TARGET} ${TARGET_EXT})
  set_target_properties(${TARGET} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}"
    INTERFACE_INCLUDE_DIRECTORIES "${INSTALL_DIR}/${INCLUDE_DIR}")
  if(LINKAGE STREQUAL "STATIC" OR LINKAGE STREQUAL "SHARED")
    set_target_properties(${TARGET} PROPERTIES
      IMPORTED_CONFIGURATIONS "Debug;Release"
      IMPORTED_LOCATION_DEBUG "${INSTALL_DIR}/${LIBRARY_DEBUG}"
      IMPORTED_LOCATION_RELEASE "${INSTALL_DIR}/${LIBRARY_RELEASE}")
  endif()
  if(LINKAGE STREQUAL "STATIC" OR LINKAGE STREQUAL "SHARED")
    set_target_properties(${TARGET} PROPERTIES
      IMPORTED_IMPLIB_DEBUG "${INSTALL_DIR}/${IMPORT_LIBRARY_DEBUG}"
      IMPORTED_IMPLIB_RELEASE "${INSTALL_DIR}/${IMPORT_LIBRARY_RELEASE}")
  endif()

  #message(STATUS
  #  "${TARGET} settings:\n"
  #  "  Definitions:\t${COMPILE_DEFINITIONS}\n"
  #  "  Include:\t\t${INSTALL_DIR}/${INCLUDE_DIR}\n"
  #  "  Library (debug):\t${INSTALL_DIR}/${LIBRARY_DEBUG}\n"
  #  "  Library (release):\t${INSTALL_DIR}/${LIBRARY_RELEASE}\n"
  #  "  Import (debug):\t${INSTALL_DIR}/${IMPORT_LIBRARY_DEBUG}\n"
  #  "  Import (release):\t${INSTALL_DIR}/${IMPORT_LIBRARY_RELEASE}")
endfunction(add_external_library)
