vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO emilk/imgui_software_renderer
    REF b5ae63a9e42eccf7db3bf64696761a53424c53dd
    SHA512 ab318b6aed050c1869dd71e467b2830ce031bb35a8310320b51c92a1fe47c13472000b078bb8cae5b251275ba383a297246c7d589fc24d171ebb6b4bb18a9cf1
    HEAD_REF master
    PATCHES
      fix-include.patch
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(
  PACKAGE_NAME imgui_software_renderer
  CONFIG_PATH lib/cmake/imgui_software_renderer
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# License
file(READ "${SOURCE_PATH}/README.md" readme_contents)
string(FIND "${readme_contents}" "## License" license_pos)
string(SUBSTRING "${readme_contents}" ${license_pos} -1 license_contents)
file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright" "${license_contents}")
