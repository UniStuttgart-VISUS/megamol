vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO LTLA/aarand
    REF e8397e9e648379f6086b20a83910c39ed8e4dd45
    SHA512 8e7c38c2ff149440982d00955b55fe61d15197a84dc74284e61c46774c1c0d49e2e2f52eff564713a6264c0c165f6baa05001c4afc8d4c04985edaabffc9ac7c
    HEAD_REF master
)

file(COPY "${SOURCE_PATH}/include/aarand" DESTINATION "${CURRENT_PACKAGES_DIR}/include/ltla")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/ltla_aarandConfig.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/ltla_aarand")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
