vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO andyborrell/imgui_tex_inspect
    REF 80ffc679e8f3f477d861d7a806e072098e94158c
    SHA512 595c13d9cc57c0357e87fb56a5e88d140cc0e377290b28c61e15e587bdafcb8c0dd755eb9301c58ebf0ce656b9875bf46687a08b4d474a7bb7442389e6e4e60f
    HEAD_REF master
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DIMGUITEXINSPECT_OPENGL_INCLUDE=GLAD2
      -DGLAD_INCLUDE_DIR="${CMAKE_CURRENT_LIST_DIR}/../../../externals/glad/include"
)

vcpkg_cmake_install()

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup()

file(INSTALL "${SOURCE_PATH}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
