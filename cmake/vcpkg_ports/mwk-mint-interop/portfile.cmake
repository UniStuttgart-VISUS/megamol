vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/MWK-mint
    REF 48339c9c2229857aba2c7f2f5f8c6f37ec5aa8e2
    SHA512 6f6380d43e2878763889b929560d272f5eb04e470df1f933f019936ce78699a558eb1e9daac2d67180059957156168663f136db69f66d6c08b8c7b06b607c437
    PATCHES
      interop_include.patch
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}/interop")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/mwk-mint-interopConfig.cmake.in" DESTINATION "${SOURCE_PATH}/interop")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/interop"
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
#file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/mwk-mint-interop)

#file(INSTALL ${SOURCE_PATH}/LICENSE.txt DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
file(WRITE ${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright "")

