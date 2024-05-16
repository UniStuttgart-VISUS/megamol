vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/dataversepp
    REF "master"
    SHA512 419646e48403e0b8c7bcefd1583bb985203088218c75df756c0ecdf2cdb64f37afc33dcd715b239fb9d73210564433827d2a8abf5310cf6cb66eea5eb5da1927
    HEAD_REF master
    PATCHES
      01_curl.patch
      02_curl.patch
      03_curl.patch
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DFETCHCONTENT_FULLY_DISCONNECTED=OFF
      -DDATAVERSE_BuildTests=OFF
      -DDATAVERSE_Unicode=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME dataversepp
    CONFIG_PATH lib/cmake/dataverse
)

vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENCE")
