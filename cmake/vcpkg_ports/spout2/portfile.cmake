#vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO leadedge/Spout2
    REF 9db0efadba16e1d884164d348f556922cfc80c50
    SHA512 d45613590fb53155c90839cf6eb7fe646ef4ec463b6cd1624aff54870818f0bc4faccded78a6b2c089fa4e8756cf15c7e17def2ef32ac6c34144e562b58c5d8b
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DSKIP_INSTALL_ALL=OFF
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/spout2)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
