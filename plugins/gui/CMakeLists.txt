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

  # GLFW
  if(USE_GLFW)
    require_external(glfw3)
  endif()

  ## ImGui
  require_external(imgui)
  set_source_files_properties(${imgui_files} PROPERTIES GENERATED TRUE)
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "resources/*")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "shaders/*")

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files} ${imgui_files} ${shader_files} ${resource_files})
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
    
  # Add C++ filesystem support, except for "stampede".
  if(UNIX)
    cmake_host_system_information(RESULT fqdn QUERY FQDN)
    if(NOT fqdn MATCHES ".*stampede2.*")
      target_link_libraries(${PROJECT_NAME} PRIVATE "stdc++fs")
      target_compile_definitions(${PROJECT_NAME} PRIVATE GUI_USE_FILESYSTEM)
    endif()
  elseif(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE GUI_USE_FILESYSTEM)
  endif()
  
  if(USE_GLFW)
      target_compile_definitions(${PROJECT_NAME} PRIVATE GUI_USE_GLFW)
  endif()   
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS IMGUI_IMPL_OPENGL_LOADER_GLAD)
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    "include" "src"
    "${SOURCE_DIR}/examples/" "${SOURCE_DIR}/misc/cpp/")
  target_link_libraries(${PROJECT_NAME} PRIVATE vislib core glad PUBLIC imgui )
  if(USE_GLFW)
      target_link_libraries(${PROJECT_NAME} PRIVATE glfw3)
  endif()   

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ DESTINATION "share/shaders")  
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
  source_group("Shaders" FILES ${shader_files})  

  # Format
  add_clang_format(${PROJECT_NAME}
    STYLE "${CMAKE_SOURCE_DIR}/.clang-format"
    FILES ${public_header_files} ${header_files} ${source_files})

endif()
