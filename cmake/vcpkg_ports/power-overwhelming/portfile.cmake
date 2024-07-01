vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/power-overwhelming
    #REF "v${VERSION}"
    REF "7143f0b11777de934046ef7bd02c1cbf82fa39b9" # master on 2024-07-01
    SHA512 cc32febef11edeb98d2192e67bba302ef4f3086f4e23572a28bb19a20ce0bf2123680bf28839433cfb98ffc541663ed8e19fd162c2a6e5fdfad6802cb1cea563
    HEAD_REF master
    PATCHES
        devendor-fetchcontent-deps.patch
)

# power-overwhelming downloads some dependencies with FetchContent.
# - adl: download here as quick workaround for creating a new port
# - nlohmann-json: use vcpkg port
# - wil: is unused (only required for poweb)

vcpkg_from_github(
    OUT_SOURCE_PATH ADL_SOURCE_PATH
    REPO GPUOpen-LibrariesAndSDKs/display-library
    REF "17.1"
    SHA512 805bc1a7f221b33955d79943833d04838b459f316c2a9ad5fa1831588b07c0bbe5975aca07c90117c10c6ff22ee12a69d5a26a75e7191eb6c40c1dccccd192af
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DPWROWG_BuildDumpSensors=OFF
      -DPWROWG_BuildTests=OFF
      -DPWROWG_CustomTinkerforgeFirmwareMajor=99
      -Dadl_SOURCE_DIR=${ADL_SOURCE_PATH}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME power_overwhelming
    CONFIG_PATH lib/cmake/power_overwhelming
)

vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENCE")
