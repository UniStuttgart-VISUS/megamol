vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/mmpld_io
    REF ad5243607b62adff6be84880c0ba9c08873161f6
    SHA512 7415a5bf9a7c3b72c529ed77fb016bcc9d5e10370fddb94a609e14c75940eb4f8966970de6da830b1bfbb92aeba755aaa7ca47a4e277429c209bb0ebd96ffeae
    HEAD_REF master
)

set(VCPKG_BUILD_TYPE "release") # header-only port

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
    PACKAGE_NAME mmpld_io
    CONFIG_PATH lib/cmake/mmpld_io
)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
