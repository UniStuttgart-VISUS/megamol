set(OPENVKL_VERSION 1.3.0)

vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO OpenVKL/openvkl
  REF v${OPENVKL_VERSION}
  SHA512 39e367b970a2d681a94346ed27cc4e61a32609e23ebc142e4a2a6414f27b0462d0f47cef99007374bdfe26d0a41c165a6a147f18ad2c3e9605a602368a555405
  HEAD_REF master
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
FEATURES
    avx     OPENVKL_ISA_AVX
    avx2    OPENVKL_ISA_AVX2
    avx512  OPENVKL_ISA_AVX512SKX
    sse4   OPENVKL_ISA_SSE4
)

# TODO openimageIO and libpng?

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS ${FEATURE_OPTIONS}
    -DBUILD_TESTING=false
    -DBUILD_EXAMPLES=false
)
vcpkg_cmake_install()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/openvkl-${OPENVKL_VERSION})

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

file(INSTALL "${SOURCE_PATH}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)