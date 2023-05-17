# MegaMol
# Copyright (c) 2022, MegaMol Dev Team
# All rights reserved.
#

include(CMakeDependentOption)

# megamol_feature_option()
#
#   Adds an option `MEGAMOL_USE_<name>` to the CMake interface. Enables the vcpkg
#   feature `<name>` and set a global compile definition `MEGAMOL_USE_<name>`.
#
# Parameters:
#   OPTION_NAME:        Name of the option, only use uppercase letters, numbers and underscore.
#   OPTION_DESCRIPTION: Description of the option.
#   OPTION_DEFAULT:     Default ON/OFF.
#   OPTION_DEPENDS:     (optional) Dependencies of the option, use like `<depends>` parameter of cmake_dependent_option().
#
function(megamol_feature_option OPTION_NAME OPTION_DESCRIPTION OPTION_DEFAULT)
  # Validate option name
  if (NOT "${OPTION_NAME}" MATCHES "^[A-Z0-9_]+$")
    message(FATAL_ERROR "Option name is only allowed to contain uppercase letters, numbers and underscore, found: ${OPTION_NAME}.")
  endif ()

  # Allow CI to override all features to default to on.
  if (MEGAMOL_ENABLE_ALL_FEATURES)
    set(OPTION_DEFAULT ON)
  endif ()

  if (${ARGC} GREATER 3)
    cmake_dependent_option(MEGAMOL_USE_${OPTION_NAME} "${OPTION_DESCRIPTION}" "${OPTION_DEFAULT}" "${ARGV3}" OFF)
  else ()
    cmake_dependent_option(MEGAMOL_USE_${OPTION_NAME} "${OPTION_DESCRIPTION}" "${OPTION_DEFAULT}" "TRUE" OFF)
  endif ()

  if (MEGAMOL_USE_${OPTION_NAME})
    # Enable vcpkg feature
    string(TOLOWER ${OPTION_NAME} OPTION_NAME_LOWER)
    string(REPLACE "_" "-" OPTION_NAME_LOWER "${OPTION_NAME_LOWER}")
    list(APPEND VCPKG_MANIFEST_FEATURES "${OPTION_NAME_LOWER}")
    set(VCPKG_MANIFEST_FEATURES "${VCPKG_MANIFEST_FEATURES}" PARENT_SCOPE)

    # add global compile definition for feature (TODO prefer a target based solution)
    add_compile_definitions(MEGAMOL_USE_${OPTION_NAME})
  endif ()
endfunction()
