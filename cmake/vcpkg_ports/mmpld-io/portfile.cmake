vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO UniStuttgart-VISUS/mmpld_io
    REF 2a6528c8b8c5ed41aa4e31f53288be7b11c58101
    SHA512 e86a2f5e55c16f6b4c32cd82920fa0dc5f9152bf08fec96dbcadc51a9751404825aecf8ce253d55aa66fb60553bc806394ec2cf25bd876c140b289e2c80c3897
    HEAD_REF master
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
