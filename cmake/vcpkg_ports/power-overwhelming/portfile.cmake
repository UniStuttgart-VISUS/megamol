vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/power-overwhelming
    REF v1.3.0
    SHA512 cd235ed14f51ccfe45c6d7e5f7da57039fd4ab1b73641261bf6a5fcca640c32f59ca22ad5e140324635fa7ac3b659505cb49dd33a26afe4bcf275489856e272a
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DFETCHCONTENT_FULLY_DISCONNECTED=OFF
      -DPWROWG_BuildDumpSensors=OFF
      -DPWROWG_BuildTests=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME power_overwhelming
    CONFIG_PATH lib/cmake/power_overwhelming
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENCE")
