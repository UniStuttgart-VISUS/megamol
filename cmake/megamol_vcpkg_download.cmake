# MegaMol
# Copyright (c) 2022, MegaMol Dev Team
# All rights reserved.
#

include(FetchContent)
mark_as_advanced(FORCE
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_QUIET
  FETCHCONTENT_UPDATES_DISCONNECTED)

# Require git for download
find_package(Git REQUIRED)

FetchContent_Declare(vcpkg-download
  GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
  GIT_TAG ${MEGAMOL_VCPKG_VERSION}
  GIT_SHALLOW TRUE
  SOURCE_DIR ${MEGAMOL_VCPKG_DIR})
FetchContent_GetProperties(vcpkg-download)
if (NOT vcpkg-download_POPULATED)
  message(STATUS "Fetch vcpkg ...")
  FetchContent_Populate(vcpkg-download)
  mark_as_advanced(FORCE
    FETCHCONTENT_SOURCE_DIR_VCPKG-DOWNLOAD
    FETCHCONTENT_UPDATES_DISCONNECTED_VCPKG-DOWNLOAD)
endif ()
