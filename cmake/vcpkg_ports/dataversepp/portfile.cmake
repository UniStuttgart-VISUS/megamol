vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/dataversepp
    REF "master"
    SHA512 65d4431dab6a4b490ef710f9e8f0a58b4cdf5d1f78bbea96f2d06c5dcfb926f0359088b820bed447a7f0879a2a5774bf7cea4c727d5400e3a2cbebf870d8953c
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
