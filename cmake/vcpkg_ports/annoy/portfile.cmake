vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO spotify/annoy
    REF 323bc69b9b8fbc1d815c9a20be245f72eed76ef9 # https://github.com/spotify/annoy/pull/632
    SHA512 918db0e3c39d87164893eecf506399032404d1eb50be6b52acdcced28684b87560301de57ba92be4140fb0d2f4c88e275d24c85eaccb6f751f50238fca07d229
    HEAD_REF master
)

set(VCPKG_BUILD_TYPE "release") # header-only port

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/annoy)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
