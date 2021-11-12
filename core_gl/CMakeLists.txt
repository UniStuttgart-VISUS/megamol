#
# MegaMol™ Core_gl
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

if(BUILD_CORE)
  project(core_gl)

  # Add externals.
  require_external(glowl)
  require_external(megamol-shader-factory)

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h" "include/*.inl")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${shader_files} ${source_files})
  target_compile_definitions(${PROJECT_NAME} PRIVATE MEGAMOLCORE_EXPORTS)
  target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "src" "include")
  target_link_libraries(${PROJECT_NAME} 
    PUBLIC core vislib_gl glowl megamol-shader-factory frontend_resources  
    PRIVATE abstract_frontend_service) # abstract render api used by MegaMolGraph

  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
      LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DYNAMICBASE:NO")
  endif()

  # Source groups in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
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
  source_group("Shaders" FILES ${shader_files})

  # Installation rules for generated files
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")

endif(BUILD_CORE)
