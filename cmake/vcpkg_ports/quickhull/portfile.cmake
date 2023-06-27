vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO akuukka/quickhull
    REF 4f65e0801b8f60c9a97da2dadbe63c2b46397694
    SHA512 8710d2630605608e66d1ce17926ecda6f19cd31659bfe56944014be8a2b09fdce9c8f6bfdf0fc22bdf06fc10562de12743390e23f22c8a81d494dec5be9bc926
    HEAD_REF master
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/quickhull)

file(WRITE ${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright "As stated in the GitHub repository: 'This implementation is 100% Public Domain.'")
