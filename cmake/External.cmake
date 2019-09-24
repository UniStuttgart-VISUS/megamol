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
  set(ARGS_ONE_VALUE GIT_TAG GIT_REPOSITORY DEBUG_SUFFIX BUILD_BYPRODUCTS)
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
  message(STATUS "${MESSAGE}")

  FetchContent_Declare(${TARGET} ${DOWNLOAD_ARGS})
  FetchContent_GetProperties(${TARGET})
  FetchContent_Populate(${TARGET})

  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_${ucName})
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_${ucName})

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
  string(REPLACE "-src" "-install" ${lcName}_INSTALL_DIR "${${lcName}_SOURCE_DIR}")

  if(IS_MULTICONFIG)
    foreach(BUILD_CONFIG Release;Debug;RelWithDebInfo)
      file(MAKE_DIRECTORY "${${lcName}_BINARY_DIR}/${BUILD_CONFIG}")

      set(SUFFIX)
      if(args_DEBUG_SUFFIX AND "${BUILD_CONFIG}" STREQUAL "Debug")
        set(SUFFIX ${args_DEBUG_SUFFIX})
      endif()

      string(REPLACE "Release" "${BUILD_CONFIG}" CONF_ARGS "${CONF_ARGS}")
      string(REPLACE "<SUFFIX>" "${SUFFIX}" CONF_ARGS "${CONF_ARGS}")

      execute_process(
        COMMAND ${CMAKE_COMMAND} "-G${CMAKE_GENERATOR}" ${GEN_ARGS} ${CONF_ARGS}
          -DCMAKE_INSTALL_PREFIX:PATH=${${lcName}_INSTALL_DIR}/${BUILD_CONFIG}
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
          ${${lcName}_SOURCE_DIR}
        WORKING_DIRECTORY "${${lcName}_BINARY_DIR}/${BUILD_CONFIG}"
        OUTPUT_QUIET
        RESULT_VARIABLE CONFIG_RESULT)
    endforeach()
  else()
    set(SUFFIX)
    if(args_DEBUG_SUFFIX AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
      set(SUFFIX ${args_DEBUG_SUFFIX})
    endif()

    string(REPLACE "<SUFFIX>" "${SUFFIX}" CONF_ARGS "${CONF_ARGS}")

    execute_process(
      COMMAND ${CMAKE_COMMAND} "-G${CMAKE_GENERATOR}" ${GEN_ARGS} ${CONF_ARGS}
        -DCMAKE_INSTALL_PREFIX:PATH=${${lcName}_INSTALL_DIR}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        ${${lcName}_SOURCE_DIR}
      WORKING_DIRECTORY "${${lcName}_BINARY_DIR}"
      OUTPUT_QUIET
      RESULT_VARIABLE CONFIG_RESULT)
  endif()

  if(NOT "${CONFIG_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Fatal error while configuring ${TARGET}.")
  endif()

  # Add command for building
  if(EXISTS "${${lcName}_BINARY_DIR}/EXTERNAL_BUILT")
    file(REMOVE "${${lcName}_BINARY_DIR}/EXTERNAL_BUILT")
  endif()

  if(IS_MULTICONFIG)
    set(BUILD_AND_INSTALL)
    set(BYPRODUCTS)
    if(args_BUILD_BYPRODUCTS)
      set(BYPRODUCTS BYPRODUCTS )
    endif()
    foreach(BUILD_CONFIG Release;Debug;RelWithDebInfo)
      set(BUILD_AND_INSTALL ${BUILD_AND_INSTALL} COMMAND ${CMAKE_COMMAND} --build ${BUILD_CONFIG} --parallel --config ${BUILD_CONFIG})
      set(BUILD_AND_INSTALL ${BUILD_AND_INSTALL} COMMAND ${CMAKE_COMMAND} --build ${BUILD_CONFIG} --target install --config ${BUILD_CONFIG})

      if(args_BUILD_BYPRODUCTS)
        set(SUFFIX)
        if(args_DEBUG_SUFFIX AND "${BUILD_CONFIG}" STREQUAL "Debug")
          set(SUFFIX ${args_DEBUG_SUFFIX})
        endif()

        string(REPLACE "<SUFFIX>" "${SUFFIX}" BYPRODUCT ${args_BUILD_BYPRODUCTS})
        string(REPLACE "<INSTALL_DIR>" "${${lcName}_INSTALL_DIR}/${BUILD_CONFIG}" BYPRODUCT ${BYPRODUCT})

        set(BYPRODUCTS ${BYPRODUCTS} ${BYPRODUCT})
      endif()
    endforeach()

    add_custom_command(OUTPUT "${${lcName}_BINARY_DIR}/EXTERNAL_BUILT"
      ${BUILD_AND_INSTALL}
      COMMAND ${CMAKE_COMMAND} -E touch \"${${lcName}_BINARY_DIR}/EXTERNAL_BUILT\"
      WORKING_DIRECTORY "${${lcName}_BINARY_DIR}"
      ${BYPRODUCTS})
  else()
    set(BYPRODUCTS)
    if(args_BUILD_BYPRODUCTS)
      set(SUFFIX)
      if(args_DEBUG_SUFFIX AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(SUFFIX ${args_DEBUG_SUFFIX})
      endif()

      string(REPLACE "<SUFFIX>" "${SUFFIX}" BYPRODUCT ${args_BUILD_BYPRODUCTS})
      string(REPLACE "<INSTALL_DIR>" "${${lcName}_INSTALL_DIR}" BYPRODUCT ${BYPRODUCT})

      set(BYPRODUCTS BYPRODUCTS ${BYPRODUCT})
    endif()

    add_custom_command(OUTPUT "${${lcName}_BINARY_DIR}/EXTERNAL_BUILT"
      COMMAND ${CMAKE_COMMAND} --build . --parallel --config ${CMAKE_BUILD_TYPE}
      COMMAND ${CMAKE_COMMAND} --build . --target install --config ${CMAKE_BUILD_TYPE}
      COMMAND ${CMAKE_COMMAND} -E touch \"${${lcName}_BINARY_DIR}/EXTERNAL_BUILT\"
      WORKING_DIRECTORY "${${lcName}_BINARY_DIR}"
      ${BYPRODUCTS})
  endif()

  # Add external target
  add_custom_target(${TARGET}_ext DEPENDS "${${lcName}_BINARY_DIR}/EXTERNAL_BUILT")
  set_target_properties(${TARGET}_ext PROPERTIES FOLDER external)

  external_set_property(${TARGET} SOURCE_DIR "${${lcName}_SOURCE_DIR}")
  external_set_property(${TARGET} INSTALL_MAIN_DIR "${${lcName}_INSTALL_DIR}")
  if(IS_MULTICONFIG)
    external_set_property(${TARGET} INSTALL_DIR "${${lcName}_INSTALL_DIR}/Release")
  else()
    external_set_property(${TARGET} INSTALL_DIR "${${lcName}_INSTALL_DIR}")
  endif()
  external_set_property(${TARGET} CONFIG_DIR "${${lcName}_INSTALL_DIR}")

  if(args_DEPENDS)
    add_dependencies(${TARGET}_ext  ${args_DEPENDS})
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
#     LIBRARY_DEBUG "Debug.dll"
#     LIBRARY_RELEASE "Release.dll"
#     LIBRARY_RELWITHDEBINFO "ReleaseWithDebInfo.dll"
#     LIBRARY_MINSIZEREL "MinSizeRel.dll"
#     IMPORT_LIBRARY_DEBUG "Debug.lib"
#     IMPORT_LIBRARY_RELEASE "Release.lib"
#     IMPORT_LIBRARY_RELWITHDEBINFO "ReleaseWithDebInfo.lib"
#     IMPORT_LIBRARY_MINSIZEREL "MinSizeRel.lib"
#     INTERFACE_LIBRARIES <external_library>*")
#
function(add_external_library TARGET LINKAGE)
  set(ARGS_ONE_VALUE PROJECT COMPILE_DEFINITIONS INCLUDE_DIR CONFIG_INCLUDE_DIR LIBRARY LIBRARY_DEBUG LIBRARY_RELEASE LIBRARY_RELWITHDEBINFO LIBRARY_MINSIZEREL IMPORT_LIBRARY IMPORT_LIBRARY_DEBUG IMPORT_LIBRARY_RELEASE IMPORT_LIBRARY_RELWITHDEBINFO IMPORT_LIBRARY_MINSIZEREL INTERFACE_LIBRARIES)
  cmake_parse_arguments(args "" "${ARGS_ONE_VALUE}" "${ARGS_MULT_VALUES}" ${ARGN})

  if(NOT args_PROJECT)
    set(args_PROJECT ${TARGET})
  endif()
  _argument_default(PROJECT "")

  # Guess library properties, unless set.
  external_get_property(${PROJECT} INSTALL_MAIN_DIR)
  external_get_property(${PROJECT} CONFIG_DIR)
  _argument_default(COMPILE_DEFINITIONS "")
  _argument_default(INCLUDE_DIR include)
  _argument_default(CONFIG_INCLUDE_DIR "")
  _argument_default(LIBRARY "NOTFOUND")
  _argument_default(LIBRARY_DEBUG "${LIBRARY}")
  _argument_default(LIBRARY_RELEASE "${LIBRARY}")
  _argument_default(LIBRARY_RELWITHDEBINFO "${LIBRARY_RELEASE}")
  _argument_default(IMPORT_LIBRARY "NOTFOUND")
  _argument_default(IMPORT_LIBRARY_DEBUG "${IMPORT_LIBRARY}")
  _argument_default(IMPORT_LIBRARY_RELEASE "${IMPORT_LIBRARY}")
  _argument_default(IMPORT_LIBRARY_RELWITHDEBINFO "${IMPORT_LIBRARY_RELEASE}")

  if(IS_MULTICONFIG)
    set(INSTALL_DIR_DEBUG ${INSTALL_MAIN_DIR}/Debug)
    set(INSTALL_DIR_RELEASE ${INSTALL_MAIN_DIR}/Release)
    set(INSTALL_DIR_RELWITHDEBINFO ${INSTALL_MAIN_DIR}/RelWithDebInfo)
  else()
    set(INSTALL_DIR_DEBUG ${INSTALL_MAIN_DIR})
    set(INSTALL_DIR_RELEASE ${INSTALL_MAIN_DIR})
    set(INSTALL_DIR_RELWITHDEBINFO ${INSTALL_MAIN_DIR})
  endif()

  # Create include directory as required by INTERFACE_INCLUDE_DIRECTORIES.
  file(MAKE_DIRECTORY "${INSTALL_DIR_RELEASE}/${INCLUDE_DIR}")

  # Add an imported library.
  add_library(${TARGET} ${LINKAGE} IMPORTED GLOBAL)
  add_dependencies(${TARGET} ${PROJECT}_ext)
  set_target_properties(${TARGET} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}"
    INTERFACE_INCLUDE_DIRECTORIES "${INSTALL_DIR_RELEASE}/${INCLUDE_DIR};${CONFIG_DIR}/${CONFIG_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${INTERFACE_LIBRARIES}")
  if(LINKAGE STREQUAL "STATIC" OR LINKAGE STREQUAL "SHARED")
    set_target_properties(${TARGET} PROPERTIES
      IMPORTED_CONFIGURATIONS "Debug;Release;RelWithDebInfo"
      IMPORTED_LOCATION_DEBUG "${INSTALL_DIR_DEBUG}/${LIBRARY_DEBUG}"
      IMPORTED_LOCATION_RELEASE "${INSTALL_DIR_RELEASE}/${LIBRARY_RELEASE}"
      IMPORTED_LOCATION_RELWITHDEBINFO "${INSTALL_DIR_RELWITHDEBINFO}/${LIBRARY_RELWITHDEBINFO}")
  endif()
  if(LINKAGE STREQUAL "SHARED")
    set_target_properties(${TARGET} PROPERTIES
      IMPORTED_IMPLIB_DEBUG "${INSTALL_DIR_DEBUG}/${IMPORT_LIBRARY_DEBUG}"
      IMPORTED_IMPLIB_RELEASE "${INSTALL_DIR_RELEASE}/${IMPORT_LIBRARY_RELEASE}"
      IMPORTED_IMPLIB_RELWITHDEBINFO "${INSTALL_DIR_RELWITHDEBINFO}/${IMPORT_LIBRARY_RELWITHDEBINFO}")
  endif()
endfunction(add_external_library)

#
# Installs external targets.
#
# install_external(TARGET libA BLib C)
#
# This is a workaround for limitations of install(TARGETS ...) [1][2].
# [1] https://gitlab.kitware.com/cmake/cmake/issues/14311
# [2] https://gitlab.kitware.com/cmake/cmake/issues/14444
#
function(install_external)
  set(ARGS_MULTI_VALUE TARGETS)
  cmake_parse_arguments(args "" "" "${ARGS_MULTI_VALUE}" ${ARGN})

  foreach(target ${args_TARGETS})
    get_target_property(IMPORTED_IMPLIB_DEBUG ${target} IMPORTED_IMPLIB_DEBUG)
    get_target_property(IMPORTED_IMPLIB_RELEASE ${target} IMPORTED_IMPLIB_RELEASE)
    get_target_property(IMPORTED_IMPLIB_RELWITHDEBINFO ${target} IMPORTED_IMPLIB_RELWITHDEBINFO)
    get_target_property(IMPORTED_IMPLIB_MINSIZEREL ${target} IMPORTED_IMPLIB_MINSIZEREL)
    get_target_property(IMPORTED_LOCATION_DEBUG ${target} IMPORTED_LOCATION_DEBUG)
    get_target_property(IMPORTED_LOCATION_RELEASE ${target} IMPORTED_LOCATION_RELEASE)
    get_target_property(IMPORTED_LOCATION_RELWITHDEBINFO ${target} IMPORTED_LOCATION_RELWITHDEBINFO)
    get_target_property(IMPORTED_LOCATION_MINSIZEREL ${target} IMPORTED_LOCATION_MINSIZEREL)
    get_target_property(INTERFACE_INCLUDE_DIRECTORIES ${target} INTERFACE_INCLUDE_DIRECTORIES)

    if(NOT IMPORTED_IMPLIB_RELWITHDEBINFO)
      set(IMPORTED_IMPLIB_RELWITHDEBINFO ${IMPORTED_IMPLIB_RELEASE})
    endif()
    if(NOT IMPORTED_IMPLIB_MINSIZEREL)
      set(IMPORTED_IMPLIB_MINSIZEREL ${IMPORTED_IMPLIB_RELEASE})
    endif()

    install(DIRECTORY ${INTERFACE_INCLUDE_DIRECTORIES}/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${target})
    if(WIN32)
      install(FILES ${IMPORTED_IMPLIB_DEBUG} DESTINATION "lib" OPTIONAL)
      install(FILES ${IMPORTED_IMPLIB_RELEASE} DESTINATION "lib" OPTIONAL)
      install(FILES ${IMPORTED_IMPLIB_RELWITHDEBINFO} DESTINATION "lib" OPTIONAL)
      install(FILES ${IMPORTED_IMPLIB_MINSIZEREL} DESTINATION "lib" OPTIONAL)
      set(TARGET_DESTINATION "bin")
    else()
      set(TARGET_DESTINATION "lib")
    endif()

    # Wildcard-based install to catch PDB files and symlinks.
    install(CODE "\
      file(GLOB DEBUG_FILES \"${IMPORTED_LOCATION_DEBUG}*\")\n \
      file(GLOB RELEASE_FILES \"${IMPORTED_LOCATION_RELEASE}*\")\n \
      file(GLOB RELWITHDEBINFO_FILES \"${IMPORTED_LOCATION_RELWITHDEBINFO}*\")\n \
      file(GLOB MINSIZEREL_FILES \"${IMPORTED_LOCATION_MINSIZEREL}*\")\n \
      file(INSTALL \${DEBUG_FILES} \${RELEASE_FILES} \${RELWITHDEBINFO_FILES} \${MINSIZEREL_FILES} DESTINATION \"${CMAKE_INSTALL_PREFIX}/${TARGET_DESTINATION}\")")
  endforeach()
endfunction(install_external)
