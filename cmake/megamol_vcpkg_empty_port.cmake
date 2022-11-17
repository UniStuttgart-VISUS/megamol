set(MEGAMOL_VCPKG_EMPTY_PORT_OVERRIDE "" CACHE STRING "';'-separated list of ports which are overridden to be empty.")
mark_as_advanced(FORCE MEGAMOL_VCPKG_EMPTY_PORT_OVERRIDE)

if (NOT "${MEGAMOL_VCPKG_EMPTY_PORT_OVERRIDE}" STREQUAL "")
  set(mm_empty_ports_dir "${CMAKE_CURRENT_BINARY_DIR}/megamol_vcpkg_empty_ports")

  # Cleanup port dir
  file(REMOVE_RECURSE "${mm_empty_ports_dir}")
  file(MAKE_DIRECTORY "${mm_empty_ports_dir}")

  # Create empty port
  foreach (portname ${MEGAMOL_VCPKG_EMPTY_PORT_OVERRIDE})
    # match format "portname[feature1,feature2]"
    if (NOT "${portname}" MATCHES "^[a-z0-9-]+(\\[[a-z0-9-]+(,[a-z0-9-]+)*\\])?$")
      message(FATAL_ERROR "Invalid portname: \"${portname}\" Required format: \"portname[feature1,feature2]\", with only lowercase chars, numbers and hyphens.")
    endif ()

    # Transform to list "portname;feature1;feature2"
    string(REPLACE "," ";" portname "${portname}")
    string(REPLACE "[" ";" portname "${portname}")
    string(REPLACE "]" "" portname "${portname}")

    # Split into portname and feature list
    set(features "${portname}")
    list(POP_FRONT features portname)

    set(feature_json "")
    if (NOT "${features}" STREQUAL "")
      set(comma "") # no comma on first entry
      foreach (feature ${features})
        string(APPEND feature_json "${comma}\"${feature}\": {\"description\": \"\"}")
        set(comma ",")
      endforeach ()

      set(feature_json ",\"features\": {${feature_json}}")
    endif ()

    # TODO This still does not set the dependencies of the port we are trying to replace, so we
    #      basically need to fetch the original vcpkg.json somehow next to the empty portfile.

    set(vcpkg_json "{\"name\": \"${portname}\", \"version-string\": \"megamol-empty-port\"${feature_json}}\n")

    file(WRITE "${mm_empty_ports_dir}/${portname}/vcpkg.json" "${vcpkg_json}")
    file(WRITE "${mm_empty_ports_dir}/${portname}/portfile.cmake" "")
  endforeach ()

  # Add separator
  string(APPEND mm_empty_ports_dir ";")
else ()
  set(mm_empty_ports_dir "")
endif ()
