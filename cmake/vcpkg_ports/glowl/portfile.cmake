vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO invor/glowl
  REF e075724a649bd1d57e464d9432556fb69be22699
  SHA512 69aff3a6a703d8906d70ce08c2d9442abe8a82e7648e5e21cd445737c64ebdfca78ed8a3242813e646bfdda2ba206596c4c1651c410c074acdcf9b672f14d340
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
