include(CMakeParseArguments)
include(ExternalProject)

function(_argument_default VARIABLE)
  if(args_${VARIABLE})
    set(${VARIABLE} "${args_${VARIABLE}}" PARENT_SCOPE)
  else()
    set(${VARIABLE} "${ARGN}" PARENT_SCOPE)
  endif()
endfunction(_argument_default)

#
# Adds an external project.
#
# See ExternalProject_Add(...) for usage.
#
function(add_external_project TARGET)
  set(ARGS_ONE_VALUE GIT_TAG)
  cmake_parse_arguments(args "" "${ARGS_ONE_VALUE}" "" ${ARGN})

  if(args_GIT_TAG)
    message(STATUS "${TARGET} tag: ${args_GIT_TAG}")
  endif()
  
  # Compile arguments for ExternalProject_Add.
  set(ARGN_EXT "${ARGN}")
  list(APPEND ARGN_EXT CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>)
  
  # Add external project.
  ExternalProject_Add(${TARGET} "${ARGN_EXT}")
endfunction(add_external_project)

#
# Adds an external library, depending on an external project.
#
# add_external_library(<target> (STATIC|SHARED|INTERFACE)
#     DEPENDS <external_project>
#     LIBRARY_DEBUG "Debug.dll"
#     LIBRARY_RELEASE "Release.dll"
#     IMPORT_LIBRARY_DEBUG "Debug.lib"
#     IMPORT_LIBRARY_RELEASE "Release.lib"
#     INTERFACE_LIBRARIES <external_library>*")
#
function(add_external_library TARGET LINKAGE)
  set(ARGS_ONE_VALUE DEPENDS COMPILE_DEFINITIONS INCLUDE_DIR LIBRARY_DEBUG LIBRARY_RELEASE IMPORT_LIBRARY_DEBUG IMPORT_LIBRARY_RELEASE INTERFACE_LIBRARIES)
  cmake_parse_arguments(args "" "${ARGS_ONE_VALUE}" "" ${ARGN})
 
   if(NOT args_DEPENDS)
     message(FATAL_ERROR "Missing external project for ${TARGET} to depend on")
   endif()
  _argument_default(DEPENDS "")
 
  # Guess library properties, unless set.
  ExternalProject_Get_Property(${DEPENDS} INSTALL_DIR)
  _argument_default(COMPILE_DEFINITIONS "")
  _argument_default(INCLUDE_DIR include)
  _argument_default(LIBRARY_DEBUG "NOTFOUND")
  _argument_default(LIBRARY_RELEASE "NOTFOUND")
  _argument_default(IMPORT_LIBRARY_DEBUG "NOTFOUND")
  _argument_default(IMPORT_LIBRARY_RELEASE "NOTFOUND")

  # Create include directory as required by INTERFACE_INCLUDE_DIRECTORIES.
  file(MAKE_DIRECTORY "${INSTALL_DIR}/${INCLUDE_DIR}")

  # Add an imported library.
  add_library(${TARGET} ${LINKAGE} IMPORTED GLOBAL)
  add_dependencies(${TARGET} ${DEPENDS})
  set_target_properties(${TARGET} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}"
    INTERFACE_INCLUDE_DIRECTORIES "${INSTALL_DIR}/${INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${INTERFACE_LIBRARIES}")
  if(LINKAGE STREQUAL "STATIC" OR LINKAGE STREQUAL "SHARED")
    set_target_properties(${TARGET} PROPERTIES
      IMPORTED_CONFIGURATIONS "Debug;Release"
      IMPORTED_LOCATION_DEBUG "${INSTALL_DIR}/${LIBRARY_DEBUG}"
      IMPORTED_LOCATION_RELEASE "${INSTALL_DIR}/${LIBRARY_RELEASE}")
  endif()
  if(LINKAGE STREQUAL "SHARED")
    set_target_properties(${TARGET} PROPERTIES
      IMPORTED_IMPLIB_DEBUG "${INSTALL_DIR}/${IMPORT_LIBRARY_DEBUG}"
      IMPORTED_IMPLIB_RELEASE "${INSTALL_DIR}/${IMPORT_LIBRARY_RELEASE}")
  endif()

  #message(STATUS
  #  "${DEPENDS} / ${TARGET} settings:\n"
  #  "  Definitions:\t${COMPILE_DEFINITIONS}\n"
  #  "  Include:\t\t${INSTALL_DIR}/${INCLUDE_DIR}\n"
  #  "  Library (debug):\t${INSTALL_DIR}/${LIBRARY_DEBUG}\n"
  #  "  Library (release):\t${INSTALL_DIR}/${LIBRARY_RELEASE}\n"
  #  "  Import (debug):\t${INSTALL_DIR}/${IMPORT_LIBRARY_DEBUG}\n"
  #  "  Import (release):\t${INSTALL_DIR}/${IMPORT_LIBRARY_RELEASE}\n"
  #  "  Interface:\t${INTERFACE_LIBRARIES}")
endfunction(add_external_library)
