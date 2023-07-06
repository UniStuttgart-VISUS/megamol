vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO lvdmaaten/bhtsne
    REF cd619e6c186b909a2d8ed26fbf0b1afec770f43d
    SHA512 de599c34083af328cd0a014e02ee8b61c839361c2b020386230a3f4ff0b3e3980e3ef93d12d52b639c53b39964c571de079b8a31fe8cf0402fdd734c568fb282
    HEAD_REF master
    PATCHES
        tsne_export.patch
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/bhtsne)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
