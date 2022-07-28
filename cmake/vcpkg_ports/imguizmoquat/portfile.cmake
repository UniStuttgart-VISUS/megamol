vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
set(IMGUIZMOQUAT_VERSION 3.0)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO BrutPitt/imGuIZMO.quat
    REF v${IMGUIZMOQUAT_VERSION}
    SHA512 0315fd29bff88854135745d166bc15218df95e0ebf93bf71e991bf6e90243b6e512c7919ffc0d05dd4a8b48f79d347f535f0006d7ab312bb5c1dfb1cdc172e54
    HEAD_REF master
    PATCHES
      generate-cmakelists.patch
      generate-cmakeconfig.patch
)

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
)
vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME imguizmoquat
    CONFIG_PATH lib/cmake/imguizmoquat
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(INSTALL "${SOURCE_PATH}/license.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
