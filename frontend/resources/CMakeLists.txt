#
# MegaMol™ Module Resources 
# Copyright 2020, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
#option(BUILD_MODULE_RESOURCES "build module resources" ON)

project(frontend_resources)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

require_external(glad)

#  # Check option dependencies: First list item is always the current build option,
#  # afterwards list the dependencies.
#  set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

file(GLOB_RECURSE header_files "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp" )
file(GLOB_RECURSE source_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Add target
add_library(${PROJECT_NAME}
  STATIC ${header_files} ${source_files}
  )
set_target_properties(${PROJECT_NAME}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  OUTPUT_NAME ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE PUBLIC glad)
  
set(INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/")
# target_include_directories(${PROJECT_NAME} INTERFACE ${INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC
  ${INCLUDE_DIRS}
  #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  #$<INSTALL_INTERFACE:include/>
)

# Grouping in Visual Studio
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
source_group("Header Files" FILES ${header_files})
source_group("Source Files" FILES ${source_files})

