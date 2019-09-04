include(FetchContent)

#
# Function for downloading and configuring external libraries, hosted on GIT servers.
#
# See CMakeExternals.cmake in the main directory for example invocations.
#
function(fetch_external NAME TARGET ACTIVATED_OPTIONS DEACTIVATED_OPTIONS HIDDEN_OPTIONS)
  string(TOLOWER ${NAME} lcName)
  string(TOUPPER ${NAME} ucName)

  # Declare content origin
  FetchContent_Declare(${lcName}_ext ${ARGN})

  # Check if already available
  FetchContent_GetProperties(${lcName}_ext)
  if(${lcName}_ext_POPULATED)
    return()
  endif()

  string(REPLACE ";" " " OPTIONS "${ARGN}")
  message(STATUS "Requesting external '${NAME}' ...")
  message(STATUS "  Options: ${OPTIONS}")

  # Configure build
  set(RESET_ACTIVATED_OPTIONS "")
  foreach(ACTIVATED_OPTION IN LISTS ACTIVATED_OPTIONS)
    if(DEFINED ${ACTIVATED_OPTION} AND NOT ${ACTIVATED_OPTION})
      list(APPEND RESET_ACTIVATED_OPTIONS ${ACTIVATED_OPTION})
      message(STATUS "  Setting temporarily: ${ACTIVATED_OPTION}=ON")
    else()
      message(STATUS "  Setting: ${ACTIVATED_OPTION}=ON")
    endif()

    set(${ACTIVATED_OPTION} ON CACHE BOOL "" FORCE)
  endforeach()

  set(RESET_DEACTIVATED_OPTIONS "")
  foreach(DEACTIVATED_OPTION IN LISTS DEACTIVATED_OPTIONS)
    if(DEFINED ${DEACTIVATED_OPTION} AND ${DEACTIVATED_OPTION})
      list(APPEND RESET_DEACTIVATED_OPTIONS ${DEACTIVATED_OPTION})
      message(STATUS "  Setting temporarily: ${DEACTIVATED_OPTION}=OFF")
    else()
      message(STATUS "  Setting: ${DEACTIVATED_OPTION}=OFF")
    endif()

    set(${DEACTIVATED_OPTION} OFF CACHE BOOL "" FORCE)
  endforeach()

  # Add project
  FetchContent_Populate(${lcName}_ext)

  set(ADDING_EXTERNAL ${lcName}_ext)
  add_subdirectory(${${lcName}_ext_SOURCE_DIR} ${${lcName}_ext_BINARY_DIR} EXCLUDE_FROM_ALL)
  unset(ADDING_EXTERNAL)

  target_include_directories(${TARGET} INTERFACE
    "$<BUILD_INTERFACE:${${lcName}_ext_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${${lcName}_ext_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${${lcName}_ext_BINARY_DIR}>"
    "$<BUILD_INTERFACE:${${lcName}_ext_BINARY_DIR}/include>")

  if(FETCHCONTENT_ADDITIONAL_INCLUDES)
    foreach(ADDITIONAL_INCLUDE IN LISTS FETCHCONTENT_ADDITIONAL_INCLUDES)
      target_include_directories(${TARGET} INTERFACE
        "$<BUILD_INTERFACE:${${lcName}_ext_SOURCE_DIR}/${ADDITIONAL_INCLUDE}>"
        "$<BUILD_INTERFACE:${${lcName}_ext_BINARY_DIR}/${ADDITIONAL_INCLUDE}>")
    endforeach()
  endif()

  if(TARGET ${TARGET})
    add_library(${NAME} ALIAS ${TARGET})
  else()
    message(FATAL_ERROR "Failed to download and configure '${NAME}'! Target '${TARGET}' not found.")
  endif()

  # Reset variables
  foreach(RESET_ACTIVATED_OPTION IN LISTS RESET_ACTIVATED_OPTIONS)
    set(${RESET_ACTIVATED_OPTION} OFF CACHE BOOL "" FORCE)
  endforeach()

  foreach(RESET_DEACTIVATED_OPTION IN LISTS RESET_DEACTIVATED_OPTIONS)
    set(${RESET_DEACTIVATED_OPTION} ON CACHE BOOL "" FORCE)
  endforeach()

  # Hide project variables
  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_${ucName}_EXT)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_${ucName}_EXT)

  foreach(ACTIVATED_OPTION IN LISTS ACTIVATED_OPTIONS)
    mark_as_advanced(FORCE ${ACTIVATED_OPTION})
  endforeach()

  foreach(DEACTIVATED_OPTION IN LISTS DEACTIVATED_OPTIONS)
    mark_as_advanced(FORCE ${DEACTIVATED_OPTION})
  endforeach()

  foreach(HIDDEN_OPTION IN LISTS HIDDEN_OPTIONS)
    mark_as_advanced(FORCE ${HIDDEN_OPTION})
  endforeach()
endfunction()

#
# Function for downloading and configuring external header-only libraries, hosted on GIT servers.
#
# See CMakeExternals.cmake in the main directory for example invocations.
#
function(fetch_external_headeronly NAME)
  string(TOLOWER ${NAME} lcName)
  string(TOUPPER ${NAME} ucName)

  # Declare content origin
  FetchContent_Declare(${lcName}_ext ${ARGN})

  # Check if already available
  FetchContent_GetProperties(${lcName}_ext)
  if(${lcName}_ext_POPULATED)
    return()
  endif()

  string(REPLACE ";" " " OPTIONS "${ARGN}")
  message(STATUS "Requesting header-only external '${NAME}' ...")
  message(STATUS "  Options: ${OPTIONS}")

  # Add project
  FetchContent_Populate(${lcName}_ext)

  add_library(${NAME} INTERFACE)
  target_include_directories(${NAME} INTERFACE
    "${${lcName}_ext_SOURCE_DIR}"
    "${${lcName}_ext_SOURCE_DIR}/include")

  # Hide project variables
  mark_as_advanced(FORCE FETCHCONTENT_SOURCE_DIR_${ucName}_EXT)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED_${ucName}_EXT)
endfunction()





### Overloaded functions ################################################################

function(message TYPE)
  string(TOUPPER ${TYPE} ucType)

  if(NOT ADDING_EXTERNAL OR ucType STREQUAL "FATAL_ERROR")
    _message(${TYPE} ${ARGN})
  endif()
endfunction()

function(add_library NAME TYPE)
  _add_library(${NAME} ${TYPE} ${ARGN})

  string(TOUPPER ${TYPE} ucType)

  if(ADDING_EXTERNAL AND NOT ucType STREQUAL "INTERFACE" AND NOT ucType STREQUAL "ALIAS")
    set_target_properties(${NAME} PROPERTIES FOLDER external)
  endif()
endfunction()

function(add_executable)
  if(NOT ADDING_EXTERNAL)
    _add_executable(${ARGN})
  endif()
endfunction()

function(add_custom_target NAME)
  if(ADDING_EXTERNAL)
    _add_custom_target(${ADDING_EXTERNAL}_${NAME} ${ARGN})
  else()
    _add_custom_target(${NAME} ${ARGN})
  endif()
endfunction()

function(add_dependencies NAME)
  if(ADDING_EXTERNAL)
    if(TARGET ${NAME})
      set(target_name ${NAME})
    else()
      set(target_name ${ADDING_EXTERNAL}_${NAME})
    endif()

    set(list_var "${ARGN}")
    set(dependencies "")

    foreach(loop_var IN LISTS list_var)
      if(TARGET ${loop_var})
        list(APPEND dependencies ${loop_var})
      else()
        list(APPEND dependencies ${ADDING_EXTERNAL}_${loop_var})
      endif()
    endforeach()

    _add_dependencies(${target_name} ${dependencies})
  else()
    _add_dependencies(${NAME} ${ARGN})
  endif()
endfunction()

function(target_link_libraries NAME)
  if(TARGET ${NAME})
    _target_link_libraries(${NAME} ${ARGN})
  elseif(NOT ADDING_EXTERNAL)
    message(FATAL_ERROR "Cannot specify link libraries for target \"example\" which is not built by this project.")
  endif()
endfunction()