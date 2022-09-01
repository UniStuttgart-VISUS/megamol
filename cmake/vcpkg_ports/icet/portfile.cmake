vcpkg_from_gitlab(
    GITLAB_URL https://gitlab.kitware.com
    OUT_SOURCE_PATH SOURCE_PATH
    REPO icet/icet
    REF 365e3695c38f5dddd6cd218b3a95dcaca45d49fc # master on 2022-08-02
    SHA512 a50524044f7be3f165da32d5e3ba01e9d894b235dda592e3f27b2c49bdcbd0c27d8a3accea36f593a38f7608022dda12b6f44675dc3e1738dc17968dcd31a6ae
    HEAD_REF master
    PATCHES
        config_install.patch
)

# TODO library seems broken with MPI off
if ("mpi" IN_LIST FEATURES)
  set(USE_MPI ON)
else()
  set(USE_MPI OFF)
endif()
if ("opengl" IN_LIST FEATURES)
  set(USE_OPENGL ON)
else()
  set(USE_OPENGL OFF)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTING=OFF
        -DICET_USE_MPI=${USE_MPI}
        -DICET_USE_OFFSCREEN_EGL=OFF
        -DICET_USE_OPENGL=${USE_OPENGL}
        -DICET_USE_OPENGL3=${USE_OPENGL}
        -DICET_USE_OSMESA=OFF
        -DICET_USE_PARICOMPRESS=OFF
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/icet)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

file(INSTALL ${SOURCE_PATH}/Copyright.txt DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
