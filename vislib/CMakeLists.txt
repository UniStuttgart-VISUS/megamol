option(BUILD_VISLIB "Build vislib" ON)

if(BUILD_VISLIB)
  project(vislib)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  add_subdirectory(glad)

  require_external(spdlog)

  # Build vislib
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  set(version_files ${CMAKE_BINARY_DIR}/version/VISlibVersionInfo.cpp)

  list(REMOVE_ITEM source_files
    "src/sys/DynamicFunctionPointer.cpp"
    "src/sys/PAMException.cpp"
    "src/sys/LinuxDaemon.cpp"
    "src/graphics/d3d/AbstractD3D11WindowImpl.cpp"
    "src/graphics/d3d/AbstractD3DAdapterInformation.cpp"
    "src/graphics/d3d/D3D9AdapterInformation.cpp"
    "src/graphics/d3d/D3D11Window.cpp"
    "src/graphics/d3d/D3DCamera.cpp"
    "src/graphics/d3d/D3DException.cpp"
    "src/graphics/d3d/D3DMatrix.cpp"
    "src/graphics/d3d/D3DPoint3D.cpp"
    "src/graphics/d3d/d3dutils.cpp"
    "src/graphics/d3d/D3DVector3D.cpp"
    "src/graphics/d3d/d3dverify.cpp"
    "src/graphics/d3d/D3DVISLogo.cpp"
    "src/graphics/gl/GlutMouseInteractionAdapter.cpp"
    )

  # Target definition
  add_library(vislib STATIC ${public_header_files} ${header_files} ${source_files} ${version_files})
  target_link_libraries(vislib PUBLIC spdlog fmt PRIVATE glad ${CMAKE_THREAD_LIBS_INIT} ${OPENGL_LIBRARIES})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src"
    "${CMAKE_CURRENT_SOURCE_DIR}/glad/include")
  target_compile_definitions(vislib PUBLIC SPDLOG_FMT_EXTERNAL)
    
  if(UNIX)
    set(CURSES_NEED_NCURSES True)
    find_package(Curses REQUIRED)
    find_package(UUID REQUIRED)
    find_package(X11 REQUIRED)
    target_link_libraries(vislib PRIVATE ${X11_LIBRARIES} ${UUID_LIBRARIES} ${CURSES_LIBRARIES})
  endif()

  # Grouping in Visual Studio
  set_target_properties(vislib PROPERTIES FOLDER base)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Version" FILES ${version_files})

  # build tests if variable VISLIB_BUILD_TESTS is defined
  option(BUILD_VISLIB_TESTS "Build vislib tests" OFF)
  mark_as_advanced(BUILD_VISLIB_TESTS)
  if(VISLIB_BUILD_TESTS)
      set(VISLIB_BUILD_TESTS ${VISLIB_BUILD_TESTS})
      add_subdirectory(tests)
  endif()

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(TARGETS vislib DESTINATION "lib" EXPORT vislib-target)
endif()
