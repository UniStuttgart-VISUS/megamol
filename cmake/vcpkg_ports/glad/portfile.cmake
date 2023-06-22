# The glad repository is a generator for the glad files and requires Python and the Python package Jinja2. We want to
# avoid these dependencies. Therefore, store a glad zipfile generated with the glad webservice directly in this port.
#
# https://gen.glad.sh/
# Settings:
#   gl      Version 4.6  Compatibility
#   glx     Version 1.4
#   vulkan  Version 1.3
#   wgl     Version 1.0
# Extensions: add all (except VK_*video*, due to missing header in glad compile)
# Options: loader
#

vcpkg_extract_source_archive(
  GLAD_DIR
  ARCHIVE "${CMAKE_CURRENT_LIST_DIR}/glad.zip"
  NO_REMOVE_ONE_LEVEL
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${GLAD_DIR}")

vcpkg_cmake_configure(SOURCE_PATH ${GLAD_DIR})

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/glad)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${CMAKE_CURRENT_LIST_DIR}/LICENSE")
