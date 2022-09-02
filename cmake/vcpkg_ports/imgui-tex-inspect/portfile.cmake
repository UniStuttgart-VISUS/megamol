vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO andyborrell/imgui_tex_inspect
    REF 80ffc679e8f3f477d861d7a806e072098e94158c
    SHA512 595c13d9cc57c0357e87fb56a5e88d140cc0e377290b28c61e15e587bdafcb8c0dd755eb9301c58ebf0ce656b9875bf46687a08b4d474a7bb7442389e6e4e60f
    HEAD_REF main
    PATCHES
        loader.patch
)

if ("opengl3-binding" IN_LIST FEATURES)
  set(USE_OPENGL3 ON)
else ()
  set(USE_OPENGL3 OFF)
endif ()

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/tex_inspect_opengl_loader.h" DESTINATION "${SOURCE_PATH}/backends")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DIMGUITEXINSPECT_USE_OPENGL3=${USE_OPENGL3}
)

vcpkg_cmake_install()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(
  PACKAGE_NAME imguitexinspect
  CONFIG_PATH lib/cmake/imguitexinspect
)

file(INSTALL "${SOURCE_PATH}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
