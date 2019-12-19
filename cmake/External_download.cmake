include(CMakeParseArguments)
include(FetchContent)

#
# Request to download from the given repository.
#
# external_download(<target>
#     GIT_REPOSITORY <git-url>
#     GIT_TAG <tag or commit>)
#
function(external_download TARGET)
  # Parse arguments
  set(ARGS_ONE_VALUE GIT_TAG GIT_REPOSITORY)
  cmake_parse_arguments(args "" "${ARGS_ONE_VALUE}" "" ${ARGN})

  # Check for local version
  external_try_get_property(${TARGET} BUILD_TYPE)
  external_try_get_property(${TARGET} GIT_REPOSITORY)
  external_try_get_property(${TARGET} GIT_TAG)

  set(AVAILABLE_VERSION)
  if(DEFINED BUILD_TYPE)
    list(APPEND AVAILABLE_VERSION ${BUILD_TYPE})
  endif()
  if(DEFINED GIT_REPOSITORY)
    list(APPEND AVAILABLE_VERSION ${GIT_REPOSITORY})
  endif()
  if(DEFINED GIT_TAG)
    list(APPEND AVAILABLE_VERSION ${GIT_TAG})
  endif()

  # Check for requested version
  if(NOT args_GIT_REPOSITORY)
    message(FATAL_ERROR "No Git repository declared as source")
  endif()

  set(MESSAGE "${TARGET}: url '${args_GIT_REPOSITORY}'")
  set(DOWNLOAD_ARGS "GIT_REPOSITORY;${args_GIT_REPOSITORY}")

  set(REQUESTED_VERSION)
  list(APPEND REQUESTED_VERSION ${CMAKE_BUILD_TYPE})
  list(APPEND REQUESTED_VERSION ${args_GIT_REPOSITORY})

  if(args_GIT_TAG)
    set(MESSAGE "${MESSAGE}, tag '${args_GIT_TAG}'")
    list(APPEND DOWNLOAD_ARGS "GIT_TAG;${args_GIT_TAG}")
    list(APPEND REQUESTED_VERSION ${args_GIT_TAG})
  endif()

  list(APPEND DOWNLOAD_ARGS "GIT_SHALLOW")

  # Download immediately if necessary
  if(AVAILABLE_VERSION STREQUAL REQUESTED_VERSION)
    message(STATUS "${MESSAGE} -- already available")
  else()
    message(STATUS "${MESSAGE}")

    FetchContent_Declare(${TARGET} ${DOWNLOAD_ARGS})
    FetchContent_GetProperties(${TARGET})
    FetchContent_Populate(${TARGET})

    string(TOLOWER "${TARGET}" lcName)
    string(TOUPPER "${TARGET}" ucName)

    mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_${ucName})
    mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_${ucName})

    # Set cached version
    external_set_property(${TARGET} BUILD_TYPE ${CMAKE_BUILD_TYPE})
    external_set_property(${TARGET} GIT_REPOSITORY ${args_GIT_REPOSITORY})

    if(args_GIT_TAG)
      external_set_property(${TARGET} GIT_TAG ${args_GIT_TAG})
    endif()

    external_set_typed_property(${TARGET} NEW_VERSION TRUE BOOL)

    # Set source and binary directory
    external_set_property(${TARGET} SOURCE_DIR "${${lcName}_SOURCE_DIR}")
    external_set_property(${TARGET} BINARY_DIR "${${lcName}_BINARY_DIR}")
  endif()
endfunction()
