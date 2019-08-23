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

  # Add ImGui.
  set(IMGUI_LIB "lib/${CMAKE_STATIC_LIBRARY_PREFIX}imgui${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_external_project(ImGui_ext
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG "v1.70"
    BUILD_BYPRODUCTS "<INSTALL_DIR>/${IMGUI_LIB}"
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ImGui/CMakeLists.txt"
      "<SOURCE_DIR>/CMakeLists.txt")
  add_external_library(ImGui STATIC
    DEPENDS ImGui_ext
    INCLUDE_DIR "src/ImGui_ext"
    LIBRARY ${IMGUI_LIB})

  ExternalProject_Get_property(ImGui_ext SOURCE_DIR)
  set(imgui_files
    "${SOURCE_DIR}/examples/imgui_impl_opengl3.cpp"
    "${SOURCE_DIR}/examples/imgui_impl_opengl3.h"
    "${SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp"
    "${SOURCE_DIR}/misc/cpp/imgui_stdlib.h")    
  set_source_files_properties(${imgui_files} PROPERTIES GENERATED TRUE)

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "resources/*")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files} ${imgui_files} ${resource_files})
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  if(UNIX)
    # Add C++ filesystem support, except for "stampede".
    cmake_host_system_information(RESULT fqdn QUERY FQDN)
    if(NOT fqdn MATCHES ".*stampede2.*")
      target_link_libraries(${PROJECT_NAME} PRIVATE "stdc++fs")
      target_compile_definitions(${PROJECT_NAME} PRIVATE GUI_USE_FILEUTILS)
    endif()
  elseif(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE GUI_USE_FILEUTILS)
  endif()
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS IMGUI_IMPL_OPENGL_LOADER_GLAD)
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    "include" "src"
    "${SOURCE_DIR}/examples/" "${SOURCE_DIR}/misc/cpp/")
  target_link_libraries(${PROJECT_NAME} PRIVATE vislib core glad PUBLIC ImGui)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    if (MSVC)
      install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
    endif()
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("ImGui Files" FILES ${imgui_files})
  source_group("Resources" FILES ${resource_files})

  # Format
  add_clang_format(${PROJECT_NAME}
    STYLE "${CMAKE_SOURCE_DIR}/.clang-format"
    FILES ${public_header_files} ${header_files} ${source_files})

endif()
