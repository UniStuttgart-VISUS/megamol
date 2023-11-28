vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/power-overwhelming
    #REF "v${VERSION}"
    REF "async"
    #SHA512 6e7d7d0057a477b62cc3c038d7707d230f48233e0ce68c479eefc1acf7893b57be8850492b362102008c670250f35f350b1a4780a176db258181cb7ba37ea0b7
    SHA512 593cf782ceb859ff677960bbf8bc8ab6994ba226efb9aaef84e2cd503b79235993df1983d55784b63093d2dc9d60782c170c2693e44314723fb5dd815b61bdfb
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DFETCHCONTENT_FULLY_DISCONNECTED=OFF
      -DPWROWG_BuildDumpSensors=OFF
      -DPWROWG_BuildTests=OFF
      -DPWROWG_CustomTinkerforgeFirmwareMajor=99
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME power_overwhelming
    CONFIG_PATH lib/cmake/power_overwhelming
)

vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENCE")
