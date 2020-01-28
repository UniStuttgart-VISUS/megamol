find_package(Git)

if(NOT GIT_FOUND)
  message(FATAL_ERROR "Git not found! This is necessary to download external dependencies.")
endif()

function(GIT_DOWNLOAD GIT_REPOSITORY GIT_TAG TARGET_DIR)
  if(EXISTS "${CMAKE_SOURCE_DIR}/${TARGET_DIR}/.git")
    return()
  endif()

  message(STATUS "Downloading external git from '${GIT_REPOSITORY}' into '${CMAKE_SOURCE_DIR}/${TARGET_DIR}'")

  # Create directory and initialize git repository, if not already done
  execute_process(
    COMMAND           ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/${TARGET_DIR}"
    COMMAND           ${GIT_EXECUTABLE} init
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/${TARGET_DIR}"
    OUTPUT_VARIABLE   git_output
    ERROR_VARIABLE    git_output
    RESULT_VARIABLE   git_failed)

  if(git_failed)
    message(FATAL_ERROR "Error creating or initializing git directory: ${git_output}")
  endif()

  # Add remote
  execute_process(
    COMMAND           ${GIT_EXECUTABLE} remote add origin ${GIT_REPOSITORY}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/${TARGET_DIR}"
    OUTPUT_VARIABLE   git_output
    ERROR_VARIABLE    git_output
    RESULT_VARIABLE   git_failed)

  if(git_failed)
    message(FATAL_ERROR "Error adding remote: ${git_output}")
  endif()

  # Fetch origin, using requested tag or commit hash
  execute_process(
    COMMAND           ${GIT_EXECUTABLE} fetch origin ${GIT_TAG}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/${TARGET_DIR}"
    OUTPUT_VARIABLE   git_output
    ERROR_VARIABLE    git_output
    RESULT_VARIABLE   git_failed)

  if(git_failed)
    message(FATAL_ERROR "Error fetching origin: ${git_output}")
  endif()

  # Reset to requested commit or tag
  execute_process(
    COMMAND           ${GIT_EXECUTABLE} reset --hard FETCH_HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/${TARGET_DIR}"
    OUTPUT_VARIABLE   git_output
    ERROR_VARIABLE    git_output
    RESULT_VARIABLE   git_failed)

  if(git_failed)
    message(FATAL_ERROR "Error getting head: ${git_output}")
  endif()
endfunction()