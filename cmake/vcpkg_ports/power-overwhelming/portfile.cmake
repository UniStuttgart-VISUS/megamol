vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/power-overwhelming
    #REF "v${VERSION}"
    REF "11e58c7c4a9b861e87e40d4345d4aeba9f67b15b" # master on 2024-06-30
    SHA512 3dc8c6d9d89c30995d07e054f703c134990e6d02caa9957cc110a5b599269cd83b50897591c3cf7a106945effe209d588c93f2b86385a2034632d892942a7644
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
