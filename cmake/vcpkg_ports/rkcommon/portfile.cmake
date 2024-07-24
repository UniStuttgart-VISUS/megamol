set(RKCOMMON_VERSION 1.10.0)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO ospray/rkcommon
    REF v${RKCOMMON_VERSION}
    SHA512 1dacc9ab0a3abe8b7b21cb2c6518fd1a59ce6008026ccb72eb712bf7859440317a288dbe87602f00ff83b9efea6b020c87c3d0b1febc5abc99d1a76fa3469f19
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DBUILD_TESTING=false
      -DINSTALL_DEPS=false
)

vcpkg_cmake_install()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/rkcommon-${RKCOMMON_VERSION})

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

file(INSTALL ${SOURCE_PATH}/LICENSE.txt DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)

