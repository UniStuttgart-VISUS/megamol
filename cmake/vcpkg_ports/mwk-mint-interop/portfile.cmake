vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/MWK-mint
    REF d46bbe0ba10cd57f21476c62908502e88552fd60
    SHA512 b9c1d7da4d28cb98e9937e65c6171779e78993cea36025d629386ed0097a6b749242a740f61c578c9dc907934297e92124ae1598f183c4e7c9d59ccadd5e76f2
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

