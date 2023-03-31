# MegaMol
# Copyright (c) 2022, MegaMol Dev Team
# All rights reserved.
#

# Notes:
# This feature allows to automatically overwrite any port with an empty port.
# The port should still keep all features and also load all dependencies. In
# order to do so, we copy the original vcpkg.json next an empty portfile.
# Currently this could only be done for ports from the upstream vcpkg repo and
# overlay ports within the MegaMol repository, because they already exist on
# well known paths within the filesystem. External repositories are not
# implemented. Further, this features just copies the available vcpkg.json
# files, ignoring all versioning features of vcpkg.

set(MEGAMOL_VCPKG_EMPTY_PORT_OVERRIDE "" CACHE STRING "Semicolon-separated list of ports which are overridden to be empty.")
mark_as_advanced(FORCE MEGAMOL_VCPKG_EMPTY_PORT_OVERRIDE)

if (NOT "${MEGAMOL_VCPKG_EMPTY_PORT_OVERRIDE}" STREQUAL "")
  set(mm_empty_ports_dir "${CMAKE_CURRENT_BINARY_DIR}/megamol_vcpkg_empty_ports")

  # Cleanup port dir
  file(REMOVE_RECURSE "${mm_empty_ports_dir}")
  file(MAKE_DIRECTORY "${mm_empty_ports_dir}")

  # Directories for searching port files
  set(port_search_dirs "${VCPKG_OVERLAY_PORTS};${MEGAMOL_VCPKG_DIR}/ports")

  # Create empty port
  foreach (portname ${MEGAMOL_VCPKG_EMPTY_PORT_OVERRIDE})
    # Search port
    unset(vcpkg_file)
    find_file(vcpkg_file "vcpkg.json" PATHS ${port_search_dirs} PATH_SUFFIXES "${portname}" NO_CACHE NO_DEFAULT_PATH)

    if (NOT vcpkg_file)
      message(FATAL_ERROR "Cannot overwrite port \"${portname}\", vcpkg.json not found.")
    endif ()

    file(COPY "${vcpkg_file}" DESTINATION "${mm_empty_ports_dir}/${portname}/")
    file(WRITE "${mm_empty_ports_dir}/${portname}/portfile.cmake" "")
    unset(vcpkg_file)
  endforeach ()

  set(VCPKG_OVERLAY_PORTS "${mm_empty_ports_dir};${VCPKG_OVERLAY_PORTS}")
endif ()
