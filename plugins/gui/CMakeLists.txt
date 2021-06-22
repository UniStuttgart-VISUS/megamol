# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(gui
  BUILD_DEFAULT ON
  DEPENDS_EXTERNALS PUBLIC
    glad
    imgui
    imguizmoquat
    glm)

if (gui_PLUGIN_ENABLED)
  # TODO tmp hack for gui to make this public, should be moved to include dir if headers should be public
  target_include_directories(gui PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

  target_link_libraries(gui PRIVATE vislib abstract_frontend_service)

  # Format
  file(GLOB_RECURSE all_cpp_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
  add_clang_format(gui
    STYLE "${CMAKE_SOURCE_DIR}/.clang-format"
    FILES ${all_cpp_files})
endif ()
