vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/megamol-shader-factory
    REF v0.9
    SHA512 bd2491cca56570c2a1f2be0d92b9240dc40c1c58a9496971bc48ba73b272c7ce9ab60765485ecf60248c3ad250b22c54770151a3bc2d409901005fad07aa6871
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME msf
    CONFIG_PATH lib/cmake/msf
)

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
