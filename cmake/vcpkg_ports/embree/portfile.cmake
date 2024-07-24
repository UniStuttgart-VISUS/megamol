# stole the embree3 port and added ispc
set(EMBREE3_VERSION 3.13.4)
vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO embree/embree
    REF v${EMBREE3_VERSION}
    SHA512 685c3935fabe1bfa7260ef148df26b686b085b75011d72011461471cbcef786a5ce7a0e85c57b2df05798489a2d4e80a8d3ee5df986029edad7df7511d99c0ca
    HEAD_REF master
    PATCHES
        fix-path.patch
        cmake_policy.patch
        fix-targets-file-not-found.patch
)

list(LENGTH FEATURES FEATURE_COUNT)
#message("FEATURE_COUNT=${FEATURE_COUNT}")

string(COMPARE EQUAL ${VCPKG_LIBRARY_LINKAGE} static EMBREE_STATIC_LIB)
string(COMPARE EQUAL ${VCPKG_CRT_LINKAGE} static EMBREE_STATIC_RUNTIME)

if (NOT VCPKG_TARGET_IS_OSX)
    if ("avx512" IN_LIST FEATURES)
        message(FATAL_ERROR "Microsoft Visual C++ Compiler does not support feature avx512 officially.")
    endif()

    vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
        FEATURES
            avx     EMBREE_ISA_AVX
            avx2    EMBREE_ISA_AVX2
            avx512  EMBREE_ISA_AVX512
            sse2    EMBREE_ISA_SSE2
            sse42   EMBREE_ISA_SSE42
    )
elseif (VCPKG_LIBRARY_LINKAGE STREQUAL static)
    list(LENGTH FEATURES FEATURE_COUNT)
    if (FEATURE_COUNT GREATER 2)
        message(WARNING [[
Using Embree as static library is not supported with AppleClang >= 9.0 when multiple ISAs are selected.
Please install embree3 with only one feature using command "./vcpkg install embree3[core,FEATURE_NAME]"
Only set feature avx automaticlly.
    ]])
        set(FEATURE_OPTIONS
            -DEMBREE_ISA_AVX=ON
            -DEMBREE_ISA_AVX2=OFF
            -DEMBREE_ISA_AVX512=OFF
            -DEMBREE_ISA_SSE2=OFF
            -DEMBREE_ISA_SSE42=OFF
        )
    endif()
endif()

set(WIN32_OPTIONS "")
if (WIN32)
  set(WIN32_OPTIONS -DEMBREE_STATIC_RUNTIME=OFF)
endif ()

if (FEATURE_COUNT LESS 2)
  message(FATAL_ERROR "You have to select at least one ISA feature!")
endif()

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS ${FEATURE_OPTIONS}
        -DEMBREE_TUTORIALS=OFF
        -DEMBREE_STATIC_LIB=OFF
        ${WIN32_OPTIONS}
)

vcpkg_cmake_install()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

if(VCPKG_LIBRARY_LINKAGE STREQUAL static)
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin" "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()
if(APPLE)
    file(REMOVE "${CURRENT_PACKAGES_DIR}/uninstall.command" "${CURRENT_PACKAGES_DIR}/debug/uninstall.command")
endif()
#file(RENAME "${CURRENT_PACKAGES_DIR}/share/doc" "${CURRENT_PACKAGES_DIR}/share/${PORT}/")

#file(COPY "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(INSTALL "${SOURCE_PATH}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)

# simple mode which does not work
# vcpkg_from_github(
#   OUT_SOURCE_PATH SOURCE_PATH
#   REPO embree/embree
#   REF 489b746c0d5010e0da10345e9dc96768bec9a037
#   SHA512 502d6ed12678df20e773da057c234e7f44b5dd0e6b8c488a9a7d2117f7c1ee8fa84aba1b8d3668caf8ebbd51337812bbb2678e71fbb76b7d311ffb723bc4f5da
#   HEAD_REF master
# )

# # TODO ISA options here
# # TODO openimageIO and libpng?

# vcpkg_cmake_configure(
#   SOURCE_PATH "${SOURCE_PATH}"
#   OPTIONS
#     -DEMBREE_TUTORIALS=false
# )
# vcpkg_cmake_install()

# vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/embree-3.13.4)

# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# file(INSTALL "${SOURCE_PATH}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)