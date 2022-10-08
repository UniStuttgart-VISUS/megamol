vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO invor/glowl
  REF e80ae434618d7a3b0056f2765dcca9d6d64c1db7
  SHA512 d2db2e5d5753e157a1b6d394a7154a2bfe4d2b8410e7e0d65cbfc437c99c66b20fbcb57753ac7eddf8836e020c0208a4f2ce9fdeb751898e3bc507e6fb90e731
  HEAD_REF master
)

if ("glm" IN_LIST FEATURES)
  set(USE_GLM ON)
else ()
  set(USE_GLM OFF)
endif ()
if ("gl-extensions" IN_LIST FEATURES)
  set(USE_GL_EXT ON)
else ()
  set(USE_GL_EXT OFF)
endif ()

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
  OPTIONS
    -DGLOWL_OPENGL_INCLUDE=GLAD2
    -DGLOWL_USE_GLM=${USE_GLM}
    -DGLOWL_USE_ARB_BINDLESS_TEXTURE=${USE_GL_EXT}
    -DGLOWL_USE_NV_MESH_SHADER=${USE_GL_EXT}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/glowl)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug" "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
