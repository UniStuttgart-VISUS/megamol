#
# MegaMol Frontend Services
# Copyright 2019, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_FRONTEND_SERVICES "build frontend services" ON)

if(BUILD_FRONTEND_SERVICES)
  project(frontend_services)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    
  require_external(glm)
  require_external(imgui)
  require_external(implot)  
  require_external(imguizmoquat)

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
#   "service_template/*.hpp"
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
    # "service_template/*.cpp"
  )

  # Remove files in gl subdirectories from source list if GL ist not enabled
  if (NOT ENABLE_GL)
    foreach(src_file ${source_files})
      string(FIND "${src_file}" "/gl/" match)
      if (NOT "${match}" STREQUAL "-1")
        message("MATCH ${src_file}")
        list(REMOVE_ITEM source_files ${src_file})
      endif()
    endforeach()
    foreach(header_file ${header_files})
      string(FIND "${header_file}" "/gl/" match)
      if (NOT "${match}" STREQUAL "-1")
        message("MATCH ${header_file}")
        list(REMOVE_ITEM header_files ${header_file})
      endif()
    endforeach()
  endif()

  file(GLOB_RECURSE gui_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "gui/*.h" "gui/*.hpp")
  file(GLOB_RECURSE gui_source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "gui/*.cpp")
  # Remove files in gl subdirectories from source list if GL ist not enabled
  if (NOT ENABLE_GL)
    foreach(src_file ${gui_source_files})
      string(FIND "${src_file}" "/gl/" match)
      if (NOT "${match}" STREQUAL "-1")
        message("MATCH ${src_file}")
        list(REMOVE_ITEM gui_source_files ${src_file})
      endif()
    endforeach()
    foreach(header_file ${gui_header_files})
      string(FIND "${header_file}" "/gl/" match)
      if (NOT "${match}" STREQUAL "-1")
        message("MATCH ${header_file}")
        list(REMOVE_ITEM gui_header_files ${header_file})
      endif()
    endforeach()
  endif()


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
    "gui/src"
    "vr_service"
   # "service_template"
  )

  add_library(abstract_frontend_service INTERFACE)
  target_include_directories(abstract_frontend_service INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
  )
  target_link_libraries(abstract_frontend_service INTERFACE frontend_resources)

  if(ENABLE_CUDA)
    find_package(CUDAToolkit)
    target_compile_definitions(${PROJECT_NAME} PUBLIC MM_CUDA_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver)
  endif()

  if(MPI_C_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_C)
  endif()
  
  # GLFW
  if (ENABLE_GL)
    require_external(glfw)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
  endif()

  if(ENABLE_VR_SERVICE_UNITY_KOLABBW)
  if(NOT ENABLE_GL)
    message( FATAL_ERROR "VR Service Unity Kolab Interop needs OpenGL enabled. Abort." )
  endif()
    require_external(mwk-mint)
    target_link_libraries(${PROJECT_NAME} PUBLIC interop Spout2)
  endif()

  # Linux Stacktrace
  # Use Boost stacktrace if installed (is fully optional).
  # sudo apt install libboost-stacktrace-dev
  #
  # Enable backtrace backend, addr2line is unusable slow and has no line numbers, unfortunately this only works for gcc out the box.
  # Clang would require backtrace.h from an installed gcc version, see
  # https://www.boost.org/doc/libs/1_77_0/doc/html/stacktrace/configuration_and_build.html#ftn.stacktrace.configuration_and_build.f3
  if (UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_package(Boost 1.71.0
      COMPONENTS
        stacktrace_basic
        stacktrace_backtrace
        stacktrace_addr2line
        stacktrace_noop)
    if (Boost_FOUND)
      target_link_libraries(${PROJECT_NAME} PUBLIC
        ${Boost_STACKTRACE_BASIC_LIBRARY}
        ${Boost_STACKTRACE_BACKTRACE_LIBRARY}
        ${Boost_STACKTRACE_ADDR2LINE_LIBRARY}
        ${Boost_STACKTRACE_NOOP_LIBRARY})
      target_compile_definitions(${PROJECT_NAME} PRIVATE
        MEGAMOL_USE_BOOST_STACKTRACE
        BOOST_STACKTRACE_USE_BACKTRACE)
    endif ()
  endif ()

  if (ENABLE_GL)
    require_external(glad)
    target_link_libraries(${PROJECT_NAME} PUBLIC core_gl glad)
  endif()

  target_link_libraries(${PROJECT_NAME} PRIVATE ${VERSION_LIBRARY} ${CMAKE_DL_LIBS} PUBLIC abstract_frontend_service core mmstd glm imgui implot imguizmoquat)


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

endif()
