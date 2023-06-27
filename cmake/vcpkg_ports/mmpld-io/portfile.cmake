vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/mmpld_io
    REF 45febfc22c326ce170bd7313149fe99c0cb73048
    SHA512 bb1d4f000bc78d2141656846b675df4bdba62d9686dc444edc6ac33bdd10f81a9f3752f9d0569773230b46232d08768b0c0956408717b3816868575daeda999a
    HEAD_REF master
)

set(VCPKG_BUILD_TYPE "release") # header-only port

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
