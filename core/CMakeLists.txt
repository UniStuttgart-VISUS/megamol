# MegaMol
# Copyright (c) 2017, MegaMol Dev Team
# All rights reserved.
#

project(core)

# Add externals.
find_package(nlohmann_json CONFIG REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(glm REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Lua REQUIRED)
find_package(imgui REQUIRED)

if (MEGAMOL_USE_CUESDK)
  find_package(CUESDK CONFIG REQUIRED)
endif ()

# Collect source files
file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h" "include/*.inl")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")

# Target definition
add_library(${PROJECT_NAME} STATIC ${public_header_files} ${header_files} ${source_files})
target_include_directories(${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  ${LUA_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    vislib
    glm::glm
    ZLIB::ZLIB
    PNG::PNG
    nlohmann_json::nlohmann_json
    ${LUA_LIBRARIES}
    frontend_resources
    OpenMP::OpenMP_CXX
    TBB::tbb
    imgui::imgui
  PRIVATE
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    abstract_frontend_service # abstract render api used by MegaMolGraph
    megamol_build_info_rc
)
if (WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE Version)
endif ()
if (MEGAMOL_USE_CUESDK)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${CUESDK_LIB})
endif ()

if (MPI_C_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_C)
endif ()

# Tracy
if (MEGAMOL_USE_TRACY)
  target_compile_definitions(${PROJECT_NAME} PUBLIC TRACY_ENABLE)
  find_package(Tracy CONFIG REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC Tracy::TracyClient)
endif ()

# Install resources
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/ DESTINATION "share/resources")

# Source groups in Visual Studio
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include/mmcore" PREFIX "Public Header Files" FILES ${public_header_files})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Header Files" FILES ${header_files})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${source_files})
