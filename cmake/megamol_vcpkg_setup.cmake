# MegaMol
# Copyright (c) 2022, MegaMol Dev Team
# All rights reserved.
#

# Allow to configure a custom vcpkg directory, otherwise download vcpkg to the
# build directory. Check for a `vcpkg` directory within the source directory
# to make offline build bundles work out of the box. By default, download only
# to the build directory, but `MEGAMOL_VCPKG_DOWNLOAD` can be used to enable
# the download for a custom directory.

set(MEGAMOL_VCPKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg" CACHE PATH "Path to vcpkg.")
set(MEGAMOL_VCPKG_DOWNLOAD OFF CACHE BOOL "Download vcpkg.")
mark_as_advanced(FORCE MEGAMOL_VCPKG_DIR MEGAMOL_VCPKG_DOWNLOAD)

if (NOT IS_DIRECTORY "${MEGAMOL_VCPKG_DIR}")
  set(MEGAMOL_VCPKG_DIR "${CMAKE_CURRENT_BINARY_DIR}/vcpkg" CACHE PATH "Path to vcpkg." FORCE)
  set(MEGAMOL_VCPKG_DOWNLOAD ON CACHE BOOL "Download vcpkg." FORCE)
endif ()

# Download vcpkg via FetchContent.
if (MEGAMOL_VCPKG_DOWNLOAD)
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
    #GIT_SHALLOW TRUE
    SOURCE_DIR ${MEGAMOL_VCPKG_DIR})
  FetchContent_GetProperties(vcpkg-download)
  if (NOT vcpkg-download_POPULATED)
    message(STATUS "Fetch vcpkg ...")
    FetchContent_Populate(vcpkg-download)
    mark_as_advanced(FORCE
      FETCHCONTENT_SOURCE_DIR_VCPKG-DOWNLOAD
      FETCHCONTENT_UPDATES_DISCONNECTED_VCPKG-DOWNLOAD)
  endif ()
endif ()

# vcpkg config
set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg_ports")
set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg_triplets") # Disable compiler tracking on Windows.
set(VCPKG_BOOTSTRAP_OPTIONS "-disableMetrics")
set(VCPKG_INSTALL_OPTIONS "--clean-after-build" "--no-print-usage")
set(CMAKE_TOOLCHAIN_FILE "${MEGAMOL_VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(ENV{VCPKG_FORCE_DOWNLOADED_BINARIES} ON) # Always download tools (i.e. CMake) to have consistent versions on all systems.

option(MEGAMOL_VCPKG_DOWNLOAD_CACHE "Download prebuilt dependency binaries if available." OFF)
if (MEGAMOL_VCPKG_DOWNLOAD_CACHE)
  set(ENV{VCPKG_BINARY_SOURCES} "clear;default,readwrite;http,https://vcpkg-cache.megamol.org/{triplet}-{name}-{sha},read")
endif ()

# Asset cache for offline build
set(MEGAMOL_VCPKG_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_assets" CACHE PATH "Path to vcpkg assets.")
mark_as_advanced(FORCE MEGAMOL_VCPKG_ASSETS_DIR)
if (MEGAMOL_VCPKG_SETUP_ASSET_CACHE)
  file(MAKE_DIRECTORY "${MEGAMOL_VCPKG_ASSETS_DIR}")
  set(ENV{X_VCPKG_ASSET_SOURCES} "x-azurl,file://${MEGAMOL_VCPKG_ASSETS_DIR}/,,readwrite")

  list(APPEND VCPKG_INSTALL_OPTIONS "--only-downloads")
  set(MEGAMOL_ENABLE_ALL_FEATURES ON)
  set(MEGAMOL_STOP_AFTER_VCPKG ON)
elseif (IS_DIRECTORY "${MEGAMOL_VCPKG_ASSETS_DIR}")
  set(ENV{X_VCPKG_ASSET_SOURCES} "x-azurl,file://${MEGAMOL_VCPKG_ASSETS_DIR}/,,read")
endif ()
