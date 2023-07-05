vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO thinks/obj-io
    REF bfe835200fdff49b45a6de4561741203f85ad028
    SHA512 b3eb413071803d26100dbfc222bfbd0f1bfbeac1c34e5d8b9abd7728f1efb95b9df36c1755d8ec58d5cee2520f8ec5f43d053ed9697184f480ed9b98f0f6ba57
    HEAD_REF master
)

set(VCPKG_BUILD_TYPE "release") # header-only port

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
