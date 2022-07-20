#
# MegaMol™ Module Resources 
# Copyright 2020, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
#option(BUILD_MODULE_RESOURCES "build module resources" ON)

project(frontend_resources)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(glm REQUIRED)
find_package(nlohmann_json REQUIRED)

if (ENABLE_CUESDK)
  require_external(CUESDK)
endif()

file(GLOB_RECURSE header_files "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp" )
file(GLOB_RECURSE source_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Add target
add_library(${PROJECT_NAME}
  STATIC ${header_files} ${source_files}
  )
set_target_properties(${PROJECT_NAME}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  OUTPUT_NAME ${PROJECT_NAME})
  
set(INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/")
# target_include_directories(${PROJECT_NAME} INTERFACE ${INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC
  ${INCLUDE_DIRS}
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  #$<INSTALL_INTERFACE:include/>
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    glm::glm
    nlohmann_json::nlohmann_json)
if (ENABLE_CUESDK)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUESDK)
endif()

# Grouping in Visual Studio
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
source_group("Header Files" FILES ${header_files})
source_group("Source Files" FILES ${source_files})
