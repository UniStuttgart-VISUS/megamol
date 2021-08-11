# MegaMol GUI
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

project(gui)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

require_external(glm)
require_external(glad)
require_external(imgui)
require_external(imguizmoquat)
  
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
      
# Install shaders
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ DESTINATION "share/shaders")
endif ()
# Install resources
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/resources")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")
endif ()

# Add target
add_library(${PROJECT_NAME} STATIC ${header_files} ${source_files})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} OUTPUT_NAME ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(${PROJECT_NAME} PRIVATE ${VERSION_LIBRARY} ${CMAKE_DL_LIBS} PUBLIC abstract_frontend_service vislib core glad glm imgui imguizmoquat)

# Grouping in Visual Studio
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
source_group("Header Files" FILES ${header_files})
source_group("Source Files" FILES ${source_files})

add_clang_format(${PROJECT_NAME}
  STYLE "${CMAKE_SOURCE_DIR}/.clang-format"
  FILES ${header_files} ${source_files})  

