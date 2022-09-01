set(TINYPLY_VERSION "2.3.4")

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO ddiakopoulos/tinyply
    REF ${TINYPLY_VERSION}
    SHA512 89d708f2293b988f69aa2c6fb90f49b91a0fc8dbc97714df84125439088df1a869beb2336939da0ea562d64eea5ec2d9f031ad02995b49b03c28e1b1213da863
    HEAD_REF master
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" SHARED_LIB)

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    OPTIONS
        -DSHARED_LIB=${SHARED_LIB}
        -DBUILD_TESTS=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/${PORT})

vcpkg_copy_pdbs()

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)

# License
file(READ "${SOURCE_PATH}/readme.md" readme_contents)
string(FIND "${readme_contents}" "License" license_line_pos)
string(SUBSTRING "${readme_contents}" ${license_line_pos} -1 license_contents)
file(WRITE ${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright "${license_contents}")

vcpkg_fixup_pkgconfig()
