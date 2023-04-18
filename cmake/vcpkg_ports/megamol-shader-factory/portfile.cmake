vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/megamol-shader-factory
    REF "v${VERSION}"
    SHA512 f74c24f43dbdac9c4204ea2ce7937d6b6d4616199091a66b4a3da8bcbd1477f2c4a01f7b386d3e8260dabec949f1a463f88174ce60b5a1ded5b65d85046960cf
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    OPTIONS
      -DMSF_INTEGRATED_GLSLANG=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME msf
    CONFIG_PATH lib/cmake/msf
)

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
