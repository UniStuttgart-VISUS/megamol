vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO chemfiles/chemfiles
    REF "${VERSION}"
    SHA512 14862234d84468ed4d1230c138cc8da49fac1ffbf98a69fb926a973147566df4ae3fd4c1ea0f925df3713588bcb51c77aa1f3d62cc2299ea9d67744bb5d138df
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/chemfiles)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
