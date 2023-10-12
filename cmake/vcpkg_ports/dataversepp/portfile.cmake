vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/dataversepp
    REF "master"
    SHA512 21c177f1b908972237193ffb8b7b0482e700dc0135fbbd2105531ff4929e576ac5321197e3de6c01bb140441bf6b536a61ad761d8bae6f3e2d5f010dc928cf4e
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
