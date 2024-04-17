vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO thinks/obj-io
    REF d8755b7d8d5120cffdfe876ee2d156d514f81f72
    SHA512 3838f73faaf6a8d51d2eb50a0b537e407e55c2e41bc65315b18f7c432c5348e65947335e65bff7c520c19f0c698e3d5d270daa244f322505e54408c35021275e
    HEAD_REF master
)

set(VCPKG_BUILD_TYPE "release") # header-only port

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
