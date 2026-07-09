vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO geringsj/imgui_tex_inspect
    REF c93070751658aad8223ac13bacc0397184bb993a # based on https://github.com/andyborrell/imgui_tex_inspect/pull/3
    SHA512 ce8a998fd1137c941ca456457041d3a60e858515ba95100e59205737ef1ec5522bbd586d57f75938802db09c7703cb06954ae0cb2643fe82e032b7f8d316ecfe
    HEAD_REF main
    PATCHES
        imgui-changes.patch
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        opengl3-binding IMGUI_TEX_INSPECT_USE_OPENGL3
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
)

vcpkg_cmake_install()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(
  PACKAGE_NAME imgui_tex_inspect
  CONFIG_PATH lib/cmake/imgui_tex_inspect
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
