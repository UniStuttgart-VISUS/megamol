#
# MegaMol™ Remote Console
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

option(BUILD_REMOTECONSOLE "Build remote Lua console" ON)

if(BUILD_REMOTECONSOLE)
  project(remoteconsole)

  if (MSVC_IDE)
    set (MSVC_TOOLSET "-${CMAKE_VS_PLATFORM_TOOLSET}")
  else ()
    set (MSVC_TOOLSET "")
  endif ()

  # include externals
  include(ExternalProject)

  if(UNIX)
    set(CXXOPTS_INSTALL_CMD "cp")
    set(CXXOPTS_INSTALL_ARGS "${PROJECT_BINARY_DIR}/extlibcxxopts-prefix/src/extlibcxxopts/include/*.hpp ${EXTERNAL_INSTALL_LOCATION}/include")
  else()
    set(CXXOPTS_INSTALL_CMD "powershell")
    set(CXXOPTS_INSTALL_ARGS "copy ${PROJECT_BINARY_DIR}/extlibcxxopts-prefix/src/extlibcxxopts/include/*.hpp ${EXTERNAL_INSTALL_LOCATION}/include")
  endif()
  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_SOURCE_DIR}/external)
  ExternalProject_Add(extlibcxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CXXOPTS_INSTALL_CMD} ${CXXOPTS_INSTALL_ARGS}
  )

  set(LIBS vislib glload lua ${EXPAT_LIBRARIES} ${ZLIB_LIBRARIES} ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${ZMQ_LIB})
  include_directories(${EXPAT_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS} ${EXTERNAL_INSTALL_LOCATION}/include)
  link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

  # Build MegaMol Remote Console
  # source files for compilation
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  include_directories(BEFORE "include" "src")

  # Target definition
  add_executable(${PROJECT_NAME} ${public_header_files} ${header_files} ${shader_files} ${source_files})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DYNAMICBASE:NO")
  endif()


  # Installation rules for generated files
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")

endif(BUILD_REMOTECONSOLE)

