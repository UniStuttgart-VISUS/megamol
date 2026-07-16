vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO geringsj/imgui_software_renderer
    REF e0c04666616da4738e0e0e3ced7aaa9fd5d47aa2
    SHA512     a68938b7a8697cb3bb677ae458998a81ff8f11331a370369908d3d3d39ccec101d278d39dd8ebfe7d36afd72e70b99cefbcb206058717c16eebd97a2c0130721
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
