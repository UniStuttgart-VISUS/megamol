vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO invor/glowl
  REF dafee75f11c5d759df30ff651d6763e4e674dd0e
  SHA512 e29524be04a5eb65c4cf41190392eb7b09632ef85cf0ede53a36f8af8a84eda3a21dde2fcce7f20a61b9a269c604b31e33537aa473e20d8385562eb62c032c82
  HEAD_REF master
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
  OPTIONS
    -DGLOWL_OPENGL_INCLUDE=GLAD2
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/glowl)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug" "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
