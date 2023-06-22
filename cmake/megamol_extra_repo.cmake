# MegaMol
# Copyright (c) 2023, MegaMol Dev Team
# All rights reserved.
#

function(megamol_extra_repo EXTRA_NAME)
  # Parse arguments
  set(optionArgs DEFAULT_ON)
  set(oneValueArgs GIT_REPO)
  set(multiValueArgs "")
  cmake_parse_arguments(MMEXTRA_ARGS "${optionArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(download_dir "${CMAKE_BINARY_DIR}/extra_repo")
  string(TOUPPER ${EXTRA_NAME} EXTRA_NAME_UPPER)
  option(MEGAMOL_${EXTRA_NAME_UPPER} "Get the MegaMol ${EXTRA_NAME} repository." "${MMEXTRA_ARGS_DEFAULT_ON}")
  if (MEGAMOL_${EXTRA_NAME_UPPER})
    set(MEGAMOL_${EXTRA_NAME_UPPER}_DIR "${download_dir}/${EXTRA_NAME}" CACHE PATH "Download directory of the ${EXTRA_NAME}.")
    option(MEGAMOL_${EXTRA_NAME_UPPER}_UPDATE "Pull updates from the ${EXTRA_NAME} repo" ON)
    if (NOT EXISTS "${MEGAMOL_${EXTRA_NAME_UPPER}_DIR}")
      message(STATUS "Downloading ${EXTRA_NAME}")
      file(MAKE_DIRECTORY "${download_dir}")
      execute_process(COMMAND
        ${GIT_EXECUTABLE} clone "${MMEXTRA_ARGS_GIT_REPO}" "${MEGAMOL_${EXTRA_NAME_UPPER}_DIR}" --depth 1
        WORKING_DIRECTORY "${download_dir}"
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    elseif (MEGAMOL_${EXTRA_NAME_UPPER}_UPDATE)
      message(STATUS "Pull ${EXTRA_NAME} updates")
      execute_process(COMMAND
        ${GIT_EXECUTABLE} pull
        WORKING_DIRECTORY "${MEGAMOL_${EXTRA_NAME_UPPER}_DIR}"
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif ()
    option(MEGAMOL_${EXTRA_NAME_UPPER}_INSTALL "Install ${EXTRA_NAME}." ON)
    if (MEGAMOL_${EXTRA_NAME_UPPER}_INSTALL)
      install(DIRECTORY "${MEGAMOL_${EXTRA_NAME_UPPER}_DIR}/" DESTINATION "${CMAKE_INSTALL_PREFIX}/${EXTRA_NAME}" PATTERN ".git" EXCLUDE)
    endif ()
  endif ()
endfunction()
