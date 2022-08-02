set(CHEMFILES_VERSION 0.10.2)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO chemfiles/chemfiles
    REF ${CHEMFILES_VERSION}
    SHA512 f6b37a0ada169e67b8bd0ea795ff72f6c942d62f39e7259e478ce5a3af7d5ceddc4f044e0d17ecf286a0ababfc3664bf73c0903febd758e6bccb34fe61fa23da
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/chemfiles)

file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)

