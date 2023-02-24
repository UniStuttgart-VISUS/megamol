vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO LTLA/powerit
    REF 5fbbe4a5fb98e75d673e47ac94a2c828f3781271
    SHA512 2283e1eca1e579b117a2ea29fcf658e039f1ba520bfb00fcfa365a01d84886cb84c9684b09a6d8a98b026c7b1c04791e46fd5d3625f69d262daf088ef4b088fc
    HEAD_REF master
)

file(COPY "${SOURCE_PATH}/include/powerit" DESTINATION "${CURRENT_PACKAGES_DIR}/include/ltla")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/ltla_poweritConfig.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/ltla_powerit")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
