vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/power-overwhelming
    REF "f84cec4e67bf8aa3974d069cf86019507535f595" # master on 2024-08-25
    SHA512 0251e5baef1c7331f18880e3d9598285d8c40f64b4f98ce4b370c0bb09416336af1d664145c40dc05c3d2692ac55b1a166b518d374c027b374052dba8de2a3bc
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

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        visa PWROWG_WithVisa
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DPWROWG_BuildDumpSensors=OFF
      -DPWROWG_BuildTests=OFF
      -DPWROWG_CustomTinkerforgeFirmwareMajor=99
      -Dadl_SOURCE_DIR=${ADL_SOURCE_PATH}
      ${FEATURE_OPTIONS}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME power_overwhelming
    CONFIG_PATH lib/cmake/power_overwhelming
)

vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENCE")
