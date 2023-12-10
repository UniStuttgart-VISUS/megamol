vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/MWK-mint
    REF ad82bed0c77c46bffda52ca737e49e2c844dc101
    SHA512  0b6f8901b97105a76b6348aa97a686c523a1a15673052bd907f36141dcd5fbbc11fdbf95c202b1352ac8a70adf4c3bfaa4a6a1737f85a301182bb427e5f9eec3
    PATCHES
      interop_include.patch
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}/interop")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/mwk-mint-interopConfig.cmake.in" DESTINATION "${SOURCE_PATH}/interop")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/interop"
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
#file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/mwk-mint-interop)

#file(INSTALL ${SOURCE_PATH}/LICENSE.txt DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
file(WRITE ${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright "")

