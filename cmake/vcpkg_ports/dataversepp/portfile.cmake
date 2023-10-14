vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/dataversepp
    REF "master"
    SHA512 457c3470c30962602e05a1f74c45448db1d521a17af9c480ae35af47a6dc9c48cfeab0fa349606d46154a5d7364a9fa29aeabf80d8d12b88a0dd4b404cb7a6e5
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
