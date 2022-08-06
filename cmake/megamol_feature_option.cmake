# MegaMol
# Copyright (c) 2022, MegaMol Dev Team
# All rights reserved.
#

function(megamol_feature_option OPTION_NAME OPTION_DESCRIPTION OPTION_DEFAULT)
  # Validate option name
  if (NOT "${OPTION_NAME}" MATCHES "^[A-Z0-9_]+$")
    message(FATAL_ERROR "Option name is only allowed to contain uppercase letters, numbers and underscore, found: ${OPTION_NAME}.")
  endif ()

  option(MEGAMOL_USE_${OPTION_NAME} "${OPTION_DESCRIPTION}" "${OPTION_DEFAULT}")

  if (MEGAMOL_USE_${OPTION_NAME})
    # Enable vcpkg feature
    string(TOLOWER ${OPTION_NAME} OPTION_NAME_LOWER)
    string(REPLACE "_" "-" OPTION_NAME_LOWER "${OPTION_NAME_LOWER}")
    list(APPEND VCPKG_MANIFEST_FEATURES "use-${OPTION_NAME_LOWER}")
    set(VCPKG_MANIFEST_FEATURES "${VCPKG_MANIFEST_FEATURES}" PARENT_SCOPE)

    # add global compile definition for feature (TODO prefer a target based solution)
    add_compile_definitions(MEGAMOL_USE_${OPTION_NAME})
  endif ()
endfunction()
