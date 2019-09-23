include(CMakeParseArguments)
include(FetchContent)

function(_argument_default VARIABLE)
  if(args_${VARIABLE})
    set(${VARIABLE} "${args_${VARIABLE}}" PARENT_SCOPE)
  else()
    set(${VARIABLE} "${ARGN}" PARENT_SCOPE)
  endif()
endfunction(_argument_default)

function(external_get_property TARGET PROPERTY)
  if(NOT DEFINED ${TARGET}_ext_${PROPERTY})
    message(FATAL_ERROR "Property ${PROPERTY} for target ${TARGET} is not defined")
  endif()

  set(${PROPERTY} ${${TARGET}_ext_${PROPERTY}} PARENT_SCOPE)
endfunction()

function(external_set_property TARGET PROPERTY VAR)
  set(${TARGET}_ext_${PROPERTY} ${VAR} CACHE STRING "" FORCE)
  mark_as_advanced(${TARGET}_ext_${PROPERTY})
endfunction()

#
# Adds an external project.
#
# See ExternalProject_Add(...) for usage.
#
function(add_external_project TARGET)
  set(ARGS_ONE_VALUE GIT_TAG GIT_REPOSITORY DEBUG_SUFFIX BUILD_BYPRODUCTS COPY)
  set(ARGS_MULT_VALUES CMAKE_ARGS PATCH_COMMAND DEPENDS)
  cmake_parse_arguments(args "" "${ARGS_ONE_VALUE}" "${ARGS_MULT_VALUES}" ${ARGN})

  string(TOLOWER "${TARGET}" lcName)
  string(TOUPPER "${TARGET}" ucName)

  # Download immediately
  if(NOT args_GIT_REPOSITORY)
    message(FATAL_ERROR "No Git repository declared as source")
  endif()
  set(MESSAGE "${TARGET}: url '${args_GIT_REPOSITORY}'")
  set(DOWNLOAD_ARGS "GIT_REPOSITORY;${args_GIT_REPOSITORY}")
  if(args_GIT_TAG)
    set(MESSAGE "${MESSAGE}, tag '${args_GIT_TAG}'")
    list(APPEND DOWNLOAD_ARGS "GIT_TAG;${args_GIT_TAG}")
  endif()
  list(APPEND DOWNLOAD_ARGS "GIT_SHALLOW")
  message(STATUS "${MESSAGE}")

  FetchContent_Declare(${TARGET} ${DOWNLOAD_ARGS})
  FetchContent_GetProperties(${TARGET})
  FetchContent_Populate(${TARGET})

  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_${ucName})
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_${ucName})

  string(REPLACE "-src" "-install" ${lcName}_INSTALL_DIR "${${lcName}_SOURCE_DIR}")

  if(NOT ${TARGET}_ext_CONFIGURED)
    # Compose arguments for external project
    set(GEN_ARGS)
    set(CONF_ARGS)

    if(CMAKE_GENERATOR_PLATFORM)
      set(GEN_ARGS ${GEN_ARGS} "-A${CMAKE_GENERATOR_PLATFORM}")
    endif()
    if(CMAKE_TOOLCHAIN_FILE)
      set(GEN_ARGS ${GEN_ARGS} -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    endif()

    if(args_CMAKE_ARGS)
      set(CONF_ARGS "${args_CMAKE_ARGS}")
    endif()

    # Apply patch
    if(args_PATCH_COMMAND)
      string(REPLACE "<SOURCE_DIR>" "${${lcName}_SOURCE_DIR}" PATCH_COMMAND "${args_PATCH_COMMAND}")

      execute_process(COMMAND ${PATCH_COMMAND} RESULT_VARIABLE CONFIG_RESULT)

      if(NOT "${CONFIG_RESULT}" STREQUAL "0")
        message(FATAL_ERROR "Fatal error while applying patch for target ${TARGET}")
      endif()
    endif()

    # Configure package
    execute_process(
      COMMAND ${CMAKE_COMMAND} "-G${CMAKE_GENERATOR}" ${GEN_ARGS} ${CONF_ARGS}
        -DCMAKE_INSTALL_PREFIX:PATH=${${lcName}_INSTALL_DIR}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_BUILD_TYPE=Release
        ${${lcName}_SOURCE_DIR}
      WORKING_DIRECTORY "${${lcName}_BINARY_DIR}"
      OUTPUT_QUIET
      RESULT_VARIABLE CONFIG_RESULT)

    if(NOT "${CONFIG_RESULT}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error while configuring ${TARGET}.")
    endif()

    set(${TARGET}_ext_CONFIGURED TRUE CACHE BOOL "" FORCE)
    mark_as_advanced(${TARGET}_ext_CONFIGURED)
  endif()

  # Add command for building
  if(NOT args_BUILD_BYPRODUCTS)
    message(FATAL_ERROR "No byproducts declared")
  endif()

  string(REPLACE "<INSTALL_DIR>" "${${lcName}_INSTALL_DIR}" BYPRODUCT ${args_BUILD_BYPRODUCTS})
  string(REPLACE "<INSTALL_DIR>" "${CMAKE_INSTALL_PREFIX}" INSTALLED_LIB ${args_BUILD_BYPRODUCTS})

  set(COPY)
  if(args_COPY)
    string(REPLACE "<INSTALL_DIR>" "${CMAKE_INSTALL_PREFIX}" COPY_ARGS ${args_COPY})
    set(COPY COMMAND ${CMAKE_COMMAND} -E copy ${COPY_ARGS})
  endif()

  add_custom_command(OUTPUT "${${lcName}_BINARY_DIR}/EXTERNAL_BUILT"
    COMMAND ${CMAKE_COMMAND} --build . --parallel --config Release
    COMMAND ${CMAKE_COMMAND} --build . --target install --config Release
    COMMAND ${CMAKE_COMMAND} -E copy \"${BYPRODUCT}\" \"${INSTALLED_LIB}\"
    ${COPY}
    COMMAND ${CMAKE_COMMAND} -E touch EXTERNAL_BUILT
    WORKING_DIRECTORY "${${lcName}_BINARY_DIR}"
    BYPRODUCTS ${BYPRODUCT})

  # Add external target
  add_custom_target(${TARGET}_ext DEPENDS "${${lcName}_BINARY_DIR}/EXTERNAL_BUILT")
  set_target_properties(${TARGET}_ext PROPERTIES FOLDER external)

  external_set_property(${TARGET} SOURCE_DIR "${${lcName}_SOURCE_DIR}")
  external_set_property(${TARGET} INSTALL_DIR "${${lcName}_INSTALL_DIR}")
  external_set_property(${TARGET} CONFIG_DIR "${${lcName}_INSTALL_DIR}")

  if(args_DEPENDS)
    add_dependencies(${TARGET}_ext ${args_DEPENDS})
  endif()

  # Create ALL target for building all external libraries at once
  if(NOT TARGET ALL_EXTERNALS)
    add_custom_target(ALL_EXTERNALS)
    set_target_properties(ALL_EXTERNALS PROPERTIES FOLDER external)
  endif()

  add_dependencies(ALL_EXTERNALS ${TARGET}_ext)
endfunction(add_external_project)

#
# Adds an external header-only project.
#
# add_external_headeronly_project(<target>
#     DEPENDS <external_projects>
#     GIT_REPOSITORY <git-url>
#     GIT_TAG <tag or commit>
#     INCLUDE_DIR <include directories relative to the source directory
#                  - omit for the source directory itself>)
#
function(add_external_headeronly_project TARGET)
  # Parse arguments
  set(ARGS_ONE_VALUE GIT_TAG GIT_REPOSITORY)
  set(ARGS_MULT_VALUES INCLUDE_DIR DEPENDS)
  cmake_parse_arguments(args "" "${ARGS_ONE_VALUE}" "${ARGS_MULT_VALUES}" ${ARGN})

  string(TOLOWER "${TARGET}" lcName)
  string(TOUPPER "${TARGET}" ucName)

  # Download immediately
  if(NOT args_GIT_REPOSITORY)
    message(FATAL_ERROR "No Git repository declared as source")
  endif()
  set(MESSAGE "${TARGET}: url '${args_GIT_REPOSITORY}'")
  set(DOWNLOAD_ARGS "GIT_REPOSITORY;${args_GIT_REPOSITORY}")
  if(args_GIT_TAG)
    set(MESSAGE "${MESSAGE}, tag '${args_GIT_TAG}'")
    list(APPEND DOWNLOAD_ARGS "GIT_TAG;${args_GIT_TAG}")
  endif()
  list(APPEND DOWNLOAD_ARGS "GIT_SHALLOW")
  message(STATUS "${MESSAGE}")

  FetchContent_Declare(${TARGET} ${DOWNLOAD_ARGS})
  FetchContent_GetProperties(${TARGET})
  FetchContent_Populate(${TARGET})

  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_${ucName})
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_${ucName})

  # Create interface library
  add_library(${TARGET} INTERFACE)

  # Add include directories
  if(args_INCLUDE_DIR)
    set(INCLUDE_DIRS)
    foreach(INCLUDE_DIR IN LISTS args_INCLUDE_DIR)
      if(EXISTS "${${lcName}_SOURCE_DIR}/${INCLUDE_DIR}")
        list(APPEND INCLUDE_DIRS "${${lcName}_SOURCE_DIR}/${INCLUDE_DIR}")
      else()
        message(WARNING "Include directory '${${lcName}_SOURCE_DIR}/${INCLUDE_DIR}' not found. Adding path '${INCLUDE_DIR}' instead.")
        list(APPEND INCLUDE_DIRS "${INCLUDE_DIR}")
      endif()
    endforeach()
  else()
    set(INCLUDE_DIRS "${${lcName}_SOURCE_DIR}")
  endif()

  target_include_directories(${TARGET} INTERFACE ${INCLUDE_DIRS})

  # Add dependencies
  if(args_DEPENDS)
    add_dependencies(${TARGET} ${args_DEPENDS})
  endif()
endfunction(add_external_headeronly_project)

#
# Adds an external library, depending on an external project.
#
# add_external_library(<target> (STATIC|SHARED|INTERFACE)
#     PROJECT <external_project>
#     LIBRARY "<library_name>.dll|so"
#     IMPORT_LIBRARY "<library_name>.lib"
#     INTERFACE_LIBRARIES <external_library>*")
#
function(add_external_library TARGET)
  set(ARGS_ONE_VALUE PROJECT COMPILE_DEFINITIONS INCLUDE_DIR CONFIG_INCLUDE_DIR LIBRARY IMPORT_LIBRARY INTERFACE_LIBRARIES)
  cmake_parse_arguments(args "" "${ARGS_ONE_VALUE}" "${ARGS_MULT_VALUES}" ${ARGN})

  if(NOT args_PROJECT)
    set(args_PROJECT ${TARGET})
  endif()
  _argument_default(PROJECT "")

  # Guess library properties, unless set.
  external_get_property(${PROJECT} INSTALL_DIR)
  external_get_property(${PROJECT} CONFIG_DIR)
  _argument_default(COMPILE_DEFINITIONS "")
  _argument_default(INCLUDE_DIR include)
  _argument_default(CONFIG_INCLUDE_DIR "")
  _argument_default(LIBRARY "NOTFOUND")
  _argument_default(IMPORT_LIBRARY "NOTFOUND")

  # Create include directory as required by INTERFACE_INCLUDE_DIRECTORIES.
  file(MAKE_DIRECTORY "${INSTALL_DIR}/${INCLUDE_DIR}")

  # Add an imported library.
  add_library(${TARGET} SHARED IMPORTED GLOBAL)
  add_dependencies(${TARGET} ${PROJECT}_ext)
  set_target_properties(${TARGET} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}"
    INTERFACE_INCLUDE_DIRECTORIES "${INSTALL_DIR}/${INCLUDE_DIR};${CONFIG_DIR}/${CONFIG_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${INTERFACE_LIBRARIES}"
    IMPORTED_CONFIGURATIONS "Release"
    IMPORTED_LOCATION "${INSTALL_DIR}/${LIBRARY}"
    IMPORTED_IMPLIB "${INSTALL_DIR}/${IMPORT_LIBRARY}")
endfunction(add_external_library)
