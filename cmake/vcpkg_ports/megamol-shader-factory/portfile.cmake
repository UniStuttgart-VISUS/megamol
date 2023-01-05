vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/megamol-shader-factory
    REF v0.10
    SHA512 8169e2b97877103194990cf2a1a5abb7e741ece9014ecf43a171e42b8ec856361cab74d7c5b56d5c562595c0e7b3900ca39b37e461a86b71d77de46692d45ef6
    HEAD_REF master
    PATCHES
      glslang.patch
)

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    OPTIONS
      -DMSF_INTEGRATED_GLSLANG=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME msf
    CONFIG_PATH lib/cmake/msf
)

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
