# Build external
if(NOT EXISTS EXTERNAL_BUILT_${CONFIG})
  message(STATUS "Building external for configuration ${CONFIG}")
  message(STATUS "INSTALL_DIR: ${INSTALL_DIR}")
  message(STATUS "INSTALL_COMMANDS: ${INSTALL_COMMANDS}")
  message(STATUS "COMMANDS: ${COMMANDS}")

  # Output based on CMake version
  set(CMD_ECHO)
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.15.0") 
    set(CMD_ECHO COMMAND_ECHO STDOUT)
  endif()

  # Build external project
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --config ${CONFIG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    ${CMD_ECHO}
    RESULT_VARIABLE BUILD_RESULT)

  if(NOT "${BUILD_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Fatal error while building external project ${TARGET}")
  endif()

  # Install external project
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --target install --config ${CONFIG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    ${CMD_ECHO}
    RESULT_VARIABLE INSTALL_RESULT)

  if(NOT "${INSTALL_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Fatal error while installing external project ${TARGET}")
  endif()

  # Execute commands (mostly for copying files)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}
    ${INSTALL_COMMANDS}
    ${COMMANDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    ${CMD_ECHO}
    RESULTS_VARIABLE COMMANDS_RESULTS)

  foreach(COMMAND_RESULT IN LISTS COMMANDS_RESULTS)
    if(NOT "${COMMAND_RESULT}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error while executing install commands for external project ${TARGET}")
    endif()
  endforeach()

  # Execute commands (mostly for copying files)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E touch EXTERNAL_BUILT_${CONFIG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE TOUCH_RESULT)

  if(NOT "${TOUCH_RESULT}" STREQUAL "0")
    message(WARNING "Could not create EXTERNAL_BUILT file for external project ${TARGET}")
  endif()
endif()
