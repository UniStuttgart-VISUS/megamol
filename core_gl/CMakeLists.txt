# MegaMol
# Copyright (c) 2017, MegaMol Dev Team
# All rights reserved.
#

project(core_gl)

# Add externals.
find_package(glowl CONFIG REQUIRED)
find_package(msf CONFIG REQUIRED)

# Collect source files
file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h" "include/*.inl")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "shaders/*")

# Target definition
add_library(${PROJECT_NAME} OBJECT ${public_header_files} ${header_files} ${source_files} ${shader_files})
target_include_directories(${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(${PROJECT_NAME}
  PUBLIC core vislib_gl glowl::glowl msf::msf_static frontend_resources OpenMP::OpenMP_CXX
  PRIVATE abstract_frontend_service) # abstract render api used by MegaMolGraph

# NVPerf
if (MEGAMOL_USE_NVPERF)
  find_package(NvPerf REQUIRED)
  find_package(NvPerfUtility REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC NvPerf-shared NvPerfUtility)
  if (WIN32)
    install(FILES "${NvPerf_SHARED_LIBRARY}" DESTINATION "${CMAKE_INSTALL_BINDIR}")
  else()
    # ${NvPerf_SHARED_LIBRARY} is a soft-link, also copy the actual file.
    get_filename_component(mm_NvPerf_LibFile "${NvPerf_SHARED_LIBRARY}" REALPATH)
    install(FILES "${NvPerf_SHARED_LIBRARY}" "${mm_NvPerf_LibFile}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
endif()

# Install shaders
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ DESTINATION "share/shaders")

# Source groups in Visual Studio
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include/mmcore_gl" PREFIX "Public Header Files" FILES ${public_header_files})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Header Files" FILES ${header_files})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${source_files})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/shaders" PREFIX "Shader Files" FILES ${shader_files})
