set (ISPC_VERSION "1.18.0")

if (APPLE)
  set(ISPC_OSSUFFIX "macOS.tar.gz")
elseif(WIN32)
  set(ISPC_OSSUFFIX "windows.zip")
  set(ISPC_SHA "97d5d7bba9933f2bcda7374c738ff8a48371487df1720b803767cc9f6fffeaf06424f713360356e7ba57ca7766a1caefe01133a5657cde59ab0bde0e35988409")
else()
  set(ISPC_OSSUFFIX "linux.tar.gz")
  set(ISPC_SHA "88f971211f069123a67da7fecdc34ebcf9a04e972928c6d6417c1d0bf4ec5eb3041d3a89df3b241d34cef16f667a9b753b9db4395fca02b75580a34de525a421")
endif()

set (ISPC_FILENAME "ispc-v${ISPC_VERSION}-${ISPC_OSSUFFIX}")

set(ISPC_URL "https://github.com/ispc/ispc/releases/download/v${ISPC_VERSION}/${ISPC_FILENAME}")

vcpkg_download_distfile(
  ISPC_ARCHIVE_PATH
  FILENAME ${ISPC_FILENAME}
  URLS ${ISPC_URL}
  SHA512 ${ISPC_SHA}
)

vcpkg_extract_source_archive(
  ISPC_DIR
  ARCHIVE ${ISPC_ARCHIVE_PATH}
)

vcpkg_copy_tools(
  SEARCH_DIR "${ISPC_DIR}/bin"
  TOOL_NAMES ispc
)
if (WIN32)
  file(INSTALL ${ISPC_DIR}/bin/ispcrt.dll DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")
endif()

set(VCPKG_POLICY_EMPTY_INCLUDE_FOLDER enabled)

file(INSTALL ${ISPC_DIR}/LICENSE.txt DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
