# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmstd_gl
  ALWAYS_BUILD
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    mmstd)

if (mmstd_gl_PLUGIN_ENABLED)
  find_package(glowl CONFIG REQUIRED)
  find_package(imgui CONFIG REQUIRED)
  find_package(imguitexinspect CONFIG REQUIRED)

  target_link_libraries(mmstd_gl
    PUBLIC
      glowl::glowl
      imgui::imgui
      imguitexinspect::imguitexinspect)

  # Additional sources
  # get include directories from compositing
  get_target_property(COMPOSITING_INCLUDE_DIRS compositing_gl INCLUDE_DIRECTORIES)
  message(${COMPOSITING_INCLUDE_DIRS})

  file(GLOB_RECURSE extra_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/*.h")
  file(GLOB_RECURSE extra_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "3rd/*.cpp")
  target_sources(mmstd_gl PRIVATE ${extra_header_files} ${extra_source_files})
  target_include_directories(mmstd_gl PRIVATE "3rd" ${COMPOSITING_INCLUDE_DIRS})
endif ()
