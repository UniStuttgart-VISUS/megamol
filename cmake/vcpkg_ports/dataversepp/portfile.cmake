vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/dataversepp
    REF 1bf1718f0e681ef01173490d50877641bd410cc2 # master on 2024-06-28
    SHA512 140c0e0882a45e8d104d426910f5710df27283fbb3e87714b3e31c08b72c4544842d35fba7b6718b9bc38377ff519eab990783dfd35ed2fc4fa8c0bc33ea3c94
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DDATAVERSE_BuildCli=OFF
      -DDATAVERSE_DownloadThirdParty=OFF
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
