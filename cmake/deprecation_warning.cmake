# MegaMol
# Copyright (c) 2023, MegaMol Dev Team
# All rights reserved.
#

set(old_option_errors "")

macro(old_option_removed opt)
  if (DEFINED "${opt}")
    list(APPEND old_option_errors "The option '${opt}' has been removed.")
  endif ()
endmacro()

macro(old_option_renamed opt_old opt_new)
  if (DEFINED "${opt_old}")
    list(APPEND old_option_errors "The option '${opt_old}' has been renamed to '${opt_new}'.")
  endif ()
endmacro()

# Removed/renamed on 2023-05-17, delete warning in ? months.
old_option_renamed(MEGAMOL_DOWNLOAD_VCPKG_CACHE MEGAMOL_VCPKG_DOWNLOAD_CACHE)
old_option_renamed(EXAMPLES MEGAMOL_EXAMPLES)
old_option_renamed(EXAMPLES_DIR MEGAMOL_EXAMPLES_DIR)
old_option_renamed(EXAMPLES_INSTALL MEGAMOL_EXAMPLES_INSTALL)
old_option_renamed(EXAMPLES_UPDATE MEGAMOL_EXAMPLES_UPDATE)
old_option_renamed(TESTS MEGAMOL_TESTS)
old_option_renamed(TESTS_DIR MEGAMOL_TESTS_DIR)
old_option_renamed(TESTS_UPDATE MEGAMOL_TESTS_UPDATE)
old_option_removed(BUILD_CORE)
old_option_removed(BUILD_FRONTEND_SERVICES)
old_option_removed(BUILD_REMOTECONSOLE)
old_option_removed(BUILD_VISLIB)

# Print error
if (NOT "${old_option_errors}" STREQUAL "")
  string(REPLACE ";" "\n" old_option_errors "${old_option_errors}")
  message(FATAL_ERROR "Outdated CMake options defined:\n${old_option_errors}")
endif ()
