vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO andyborrell/imgui_tex_inspect
    REF ccff03b844cc9845cc3e3c6ef69026fe7051d330 # https://github.com/andyborrell/imgui_tex_inspect/pull/3
    SHA512 16e73a68bb8c4473bb8b3b1ddbab62453b36f4e4d0deb5f62c209c7b05bbe12173420dee8441e3d1ab9cec25f6dd77f8f4039c1a6f91fe972c88e5c21cfa634f
    HEAD_REF main
)

if ("opengl3-binding" IN_LIST FEATURES)
  set(USE_OPENGL3 ON)
else ()
  set(USE_OPENGL3 OFF)
endif ()

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DIMGUI_TEX_INSPECT_USE_OPENGL3=${USE_OPENGL3}
)

vcpkg_cmake_install()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(
  PACKAGE_NAME imgui_tex_inspect
  CONFIG_PATH lib/cmake/imgui_tex_inspect
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(INSTALL "${SOURCE_PATH}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
