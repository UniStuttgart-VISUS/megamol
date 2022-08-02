
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO lvdmaaten/bhtsne
    REF 36b169c88250d0afe51828448dfdeeaa508f13bc
    SHA512 f16e9c9a56a36285e5b22129f9fc6fae13956cf673ea31d9ffedb0bd7fbd3b63b92ae296f3ae4af7ebe1a690875dde18e8a46c78942823ebd8dac74329257ccc
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

file(INSTALL ${CMAKE_CURRENT_LIST_DIR}/bhtsneConfig.cmake DESTINATION "${CURRENT_PACKAGES_DIR}/share/bhtsne")
file(INSTALL ${SOURCE_PATH}/LICENSE.txt DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)

