#
# MegaMol™ gui Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER gui EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build GUI" ON)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(gui)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "resources/*")

  require_external(glm)
  require_external(glad)
  require_external(imgui)
  require_external(imguizmoquat)

  # target definition
  set_source_files_properties(${imgui_files} PROPERTIES GENERATED TRUE)
  add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files} ${imgui_files} ${resource_files})

  if(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
  endif()

  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS IMGUI_IMPL_OPENGL_LOADER_GLAD)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src"
    "${SOURCE_DIR}/examples/" "${SOURCE_DIR}/misc/cpp/") # imgui
  target_link_libraries(${PROJECT_NAME} PRIVATE vislib core glad abstract_frontend_service PUBLIC imgui imguizmoquat)

  # Installation rules for generated files
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")
  if(WIN32)
    #install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    #install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    #install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  foreach(FILE_NAME ${source_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^src" "Source Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  foreach(FILE_NAME ${header_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^src" "Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  foreach(FILE_NAME ${public_header_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^include\\\\mmcore" "Public Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  source_group("Version" FILES ${version_files})
  source_group("ImGui Files" FILES ${imgui_files})
  source_group("Resources" FILES ${resource_files})

  # Format
  add_clang_format(${PROJECT_NAME}
    STYLE "${CMAKE_SOURCE_DIR}/.clang-format"
    FILES ${public_header_files} ${header_files} ${source_files})

  # Register plugin
  megamol_register_plugin(${PROJECT_NAME})
endif()
