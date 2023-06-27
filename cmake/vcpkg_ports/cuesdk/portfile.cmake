vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

if (APPLE)
  #set(CUESDK_FILE_EXT "dmg")
  #set(CUESDK_SHA "88f971211f069123a67da7fecdc34ebcf9a04e972928c6d6417c1d0bf4ec5eb3041d3a89df3b241d34cef16f667a9b753b9db4395fca02b75580a34de525a421")
elseif(WIN32)
  set(CUESDK_FILE_EXT "zip")
  set(CUESDK_SHA "64e21cb372c5d9806edb06f30e687cf7f5b540660f34ae1e42703f2c6e1f10d61d461d036bfff7a3a169b60f7a875aa45964fadfef4ae18b48c26b61d650c516")
else()
  #set(CUESDK_FILE_EXT "tar.gz")
  #set(CUESDK_SHA "88f971211f069123a67da7fecdc34ebcf9a04e972928c6d6417c1d0bf4ec5eb3041d3a89df3b241d34cef16f667a9b753b9db4395fca02b75580a34de525a421")
endif()

set (CUESDK_FILENAME "CUESDK_${VERSION}.${CUESDK_FILE_EXT}")

set(CUESDK_URL "https://github.com/CorsairOfficial/cue-sdk/releases/download/v${VERSION}/${CUESDK_FILENAME}")

vcpkg_download_distfile(
  CUESDK_ARCHIVE_PATH
  FILENAME ${CUESDK_FILENAME}
  URLS ${CUESDK_URL}
  SHA512 ${CUESDK_SHA}
)

vcpkg_extract_source_archive(
  CUESDK_DIR
  ARCHIVE ${CUESDK_ARCHIVE_PATH}
)

set(CUESDK_HEADERS
  ${CUESDK_DIR}/include/CorsairKeyIdEnum.h
  ${CUESDK_DIR}/include/CorsairLedIdEnum.h
  ${CUESDK_DIR}/include/CUESDK.h
  ${CUESDK_DIR}/include/CUESDKGlobal.h
)

if (WIN32)
  set(CUESDK_LIBS
    ${CUESDK_DIR}/lib/x64/CUESDK.x64_2015.lib
    ${CUESDK_DIR}/lib/x64/CUESDK.x64_2017.lib
    ${CUESDK_DIR}/lib/x64/CUESDK.x64_2019.lib
  )
  set(CUESDK_DLLS
    ${CUESDK_DIR}/redist/x64/CUESDK.x64_2015.dll
    ${CUESDK_DIR}/redist/x64/CUESDK.x64_2017.dll
    ${CUESDK_DIR}/redist/x64/CUESDK.x64_2019.dll
  )
  file(INSTALL ${CUESDK_HEADERS} DESTINATION "${CURRENT_PACKAGES_DIR}/include")
  file(INSTALL ${CUESDK_LIBS} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
  file(INSTALL ${CUESDK_LIBS} DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
  file(INSTALL ${CUESDK_DLLS} DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
  file(INSTALL ${CUESDK_DLLS} DESTINATION "${CURRENT_PACKAGES_DIR}/debug/bin")
  file(INSTALL ${CMAKE_CURRENT_LIST_DIR}/CUESDKConfig.cmake DESTINATION "${CURRENT_PACKAGES_DIR}/share/cuesdk")
endif()

vcpkg_fixup_pkgconfig()

file(WRITE ${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright "proprietary")
