vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/dataversepp
    REF 33993c001098526efc48bd6e7ef015255a8da338 # master on 2024-06-28
    SHA512 de4e18f354f5ed0ae6eb2cccff1420ded70559bc1229d795b6f26146a7b1cad4c32f554749bf95bf87c749eecc4a36254af44bd802012a7d4a64fbb4c301629a
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
