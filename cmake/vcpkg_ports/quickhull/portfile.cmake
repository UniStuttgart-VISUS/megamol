vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO akuukka/quickhull
    REF 9f81d1ec984a13e3596d02f61c6523e5f2971e82
    SHA512 5e7e4e4ee0726b3702f839dc4f9ad06411167b74795dcc8a254f3b97d80eaa44204b6a266d3d55e37f320b1dd6b9c8795e5423d6187242cc4c30b14a127636cc
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
