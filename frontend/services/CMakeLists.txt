# MegaMol
# Copyright (c) 2019, MegaMol Dev Team
# All rights reserved.
#

project(frontend_services)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imgui_software_renderer CONFIG REQUIRED)
find_package(imguizmoquat CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(cppzmq REQUIRED)
find_package(sol2 CONFIG REQUIRED)

file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "include/*.h"
  "include/*.hpp"
  "command/*.h"
  "command/*.hpp"
  "cuda/*h"
  "cuda/*hpp"
  "opengl_glfw/*.h"
  "opengl_glfw/*.hpp"
  "service_collection/*.hpp"
  "lua_service_wrapper/*.hpp"
  "screenshot_service/*.hpp"
  "framestatistics_service/*.hpp"
  "project_loader/*.hpp"
  "image_presentation/*.hpp"
  "remote_service/*.hpp"
  "profiling_service/*.hpp"
  "vr_service/*.hpp"
  "runtimeinfo_service/*.hpp"
  "power_service/*.hpp"
  # "service_template/*.hpp"
)

file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "command/*.cpp"
  "cuda/*.cpp"
  "opengl_glfw/*.cpp"
  "service_collection/*.cpp"
  "lua_service_wrapper/*.cpp"
  "screenshot_service/*.cpp"
  "framestatistics_service/*.cpp"
  "project_loader/*.cpp"
  "image_presentation/*.cpp"
  "remote_service/*.cpp"
  "profiling_service/*.cpp"
  "vr_service/*.cpp"
  "runtimeinfo_service/*.cpp"
  "power_service/*.cpp"
  # "service_template/*.cpp"
)

# Remove files in gl subdirectories from source list if GL ist not enabled
if (NOT MEGAMOL_USE_OPENGL)
  foreach (src_file ${source_files})
    string(FIND "${src_file}" "/gl/" match)
    if (NOT "${match}" STREQUAL "-1")
      message("MATCH ${src_file}")
      list(REMOVE_ITEM source_files ${src_file})
    endif ()
  endforeach ()
  foreach (header_file ${header_files})
    string(FIND "${header_file}" "/gl/" match)
    if (NOT "${match}" STREQUAL "-1")
      message("MATCH ${header_file}")
      list(REMOVE_ITEM header_files ${header_file})
    endif ()
  endforeach ()
endif ()

file(GLOB_RECURSE gui_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "gui/*.h" "gui/*.hpp")
file(GLOB_RECURSE gui_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "gui/*.cpp")
# Remove files in gl subdirectories from source list if GL ist not enabled
if (NOT MEGAMOL_USE_OPENGL)
  foreach (src_file ${gui_source_files})
    string(FIND "${src_file}" "/gl/" match)
    if (NOT "${match}" STREQUAL "-1")
      message("MATCH ${src_file}")
      list(REMOVE_ITEM gui_source_files ${src_file})
    endif ()
  endforeach ()
  foreach (header_file ${gui_header_files})
    string(FIND "${header_file}" "/gl/" match)
    if (NOT "${match}" STREQUAL "-1")
      message("MATCH ${header_file}")
      list(REMOVE_ITEM gui_header_files ${header_file})
    endif ()
  endforeach ()
endif ()


# Add target
add_library(${PROJECT_NAME} STATIC ${header_files} ${source_files} ${gui_header_files} ${gui_source_files})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} OUTPUT_NAME ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
  "include"
  "command"
  "cuda"
  "opengl_glfw"
  "service_collection"
  "lua_service_wrapper"
  "screenshot_service"
  "framestatistics_service"
  "project_loader"
  "image_presentation"
  "remote_service"
  "profiling_service"
  "gui/3rd"
  "gui/src"
  "vr_service"
  "runtimeinfo_service"
  "power_service"
  # "service_template"
)

add_library(abstract_frontend_service INTERFACE)
target_include_directories(abstract_frontend_service INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
)
target_link_libraries(abstract_frontend_service INTERFACE frontend_resources)

# sol2
target_link_libraries(${PROJECT_NAME} INTERFACE sol2)
target_compile_definitions(${PROJECT_NAME} PUBLIC SOL_ALL_SAFETIES_ON=1)

if (MEGAMOL_USE_CUDA)
  find_package(CUDAToolkit)
  target_compile_definitions(${PROJECT_NAME} PUBLIC MM_CUDA_ENABLED)
  target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver)
endif ()

if (MPI_C_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_C)
endif ()

if (MEGAMOL_USE_OPENGL)
  find_package(glfw3 CONFIG REQUIRED)
  find_package(glad REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE glfw PUBLIC glad::glad core_gl)
endif ()

if (MEGAMOL_USE_POWER)
  find_package(Parquet CONFIG REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>")
  find_package(power_overwhelming CONFIG REQUIRED)
  find_package(dataverse CONFIG REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC power_overwhelming::power_overwhelming dataverse::dataverse)
  if (WIN32)
    find_package(wil REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC WIL::WIL)
  endif()
endif ()

if (MEGAMOL_USE_VR_INTEROP)
  find_package(mwk-mint-interop)
  target_link_libraries(${PROJECT_NAME} PUBLIC mwk-mint-interop::mwk-mint-interop)
endif ()

# Boost.Stacktrace
if (MEGAMOL_USE_STACKTRACE)
  set(Boost_NO_WARN_NEW_VERSIONS 1)
  if (WIN32)
    find_package(Boost REQUIRED COMPONENTS stacktrace_windbg)
    target_link_libraries(${PROJECT_NAME} PUBLIC Boost::stacktrace_windbg)
  else ()
    find_package(Boost REQUIRED COMPONENTS stacktrace_backtrace)
    target_link_libraries(${PROJECT_NAME} PRIVATE Boost::stacktrace_backtrace backtrace)
    target_compile_definitions(${PROJECT_NAME} PRIVATE BOOST_STACKTRACE_USE_BACKTRACE)
  endif ()
endif ()

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    ${VERSION_LIBRARY}
    ${CMAKE_DL_LIBS}
  PUBLIC
    abstract_frontend_service
    core
    glm::glm
    imgui::imgui
    imgui_software_renderer::imgui_software_renderer
    imguizmoquat::imguizmoquat
    implot::implot
    libzmq
    cppzmq)

# Install gui resources
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gui/resources")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gui/resources/ DESTINATION "share/resources")
endif ()

# Grouping in Visual Studio
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
source_group("Header Files" FILES ${header_files})
source_group("Source Files" FILES ${source_files})
source_group("Header Files\\gui" FILES ${gui_header_files})
source_group("Source Files\\gui" FILES ${gui_source_files})
