vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO braunms/imGuIZMO.quat
    REF 9841b964a42863e31e3448f12b913601c1347d07
    SHA512 249f80f515ce8beafad88b58cc7fddf9a90b90a56a74574cff539724218c4adf278ed80ab9563ae8b54fc03a9efec7041202ce7d6af8959a879275c9659796a3
    HEAD_REF master
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/imguizmoquatConfig.cmake.in" DESTINATION "${SOURCE_PATH}")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
)
vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME imguizmoquat
    CONFIG_PATH lib/cmake/imguizmoquat
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/license.txt")
