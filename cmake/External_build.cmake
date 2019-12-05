# Build external
if(NOT EXISTS EXTERNAL_BUILT_${CONFIG})
  message(STATUS "Building external for configuration ${CONFIG}")
  message(STATUS "INSTALL_DIR: ${INSTALL_DIR}")
  message(STATUS "INSTALL_COMMANDS: ${INSTALL_COMMANDS}")
  message(STATUS "COMMANDS: ${COMMANDS}")

  execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --verbose --parallel --config ${CONFIG}
    COMMAND ${CMAKE_COMMAND} --build . --target install --config ${CONFIG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE BUILD_RESULT)

  if(NOT "${BUILD_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Fatal error while building external project ${TARGET}")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}
    ${INSTALL_COMMANDS}
    ${COMMANDS}
    COMMAND ${CMAKE_COMMAND} -E touch EXTERNAL_BUILT_${CONFIG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE INSTALL_RESULT)

  if(NOT "${INSTALL_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Fatal error while installing external project ${TARGET}")
  endif()
endif()
