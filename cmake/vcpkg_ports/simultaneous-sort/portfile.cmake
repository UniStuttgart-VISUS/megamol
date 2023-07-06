vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO alexstraub1990/simultaneous-sort
    REF 220fdf37fec2d9d3e3f7674194544ee70eb93ee7
    SHA512 96c09749a8115c8eabcfba9dfb6e082a764a84ebf42a5ae0a81c11765ce36e251f7748387a4bb6d8b8a859d2aa3dc9041b0635d1a29d00923d639c0b09cdb65b
    HEAD_REF master
)

set(VCPKG_BUILD_TYPE "release") # header-only port

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS ${FEATURE_OPTIONS}
      -Dss_option_build_test=false
)
vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
