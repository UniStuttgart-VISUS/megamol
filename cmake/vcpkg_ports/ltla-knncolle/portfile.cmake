vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO LTLA/knncolle
    REF 560436d142d08eb505fe63fb7eac12ed46fda345
    SHA512 a6686f93969d5cb4cbc546c2508edd0bc0dbf8026564f86bd322b2412ff050af7f3f547c0106d14b82f4fc16d07367aac78df1755f7090fddedf22f1618975ac
    HEAD_REF master
)

file(COPY "${SOURCE_PATH}/include/knncolle" DESTINATION "${CURRENT_PACKAGES_DIR}/include/ltla")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/ltla_knncolleConfig.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/ltla_knncolle")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
