option(BUILD_VISLIB "Build vislib" ON)


if(BUILD_VISLIB)
  project(vislib)
  set(BUILDING_VISLIB_PROJECT 1)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  set(VISLIB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "vislib include directories")
  mark_as_advanced(VISLIB_INCLUDE_DIRS)
  
  if(UNIX)
    set(CURSES_NEED_NCURSES True)
    find_package(Curses REQUIRED)
    find_package(UUID REQUIRED)
    find_package(X11 REQUIRED)
  endif()

  set(LIBS ${CMAKE_THREAD_LIBS_INIT} ${X11_LIBRARIES} -lrt -lcrypt ${UUID_LIBRARIES} ${OPENGL_LIBRARIES} ${CURSES_LIBRARIES})
  include_directories(${OPENGL_INCLUDE_DIR})


  # Build glload libraries
  add_subdirectory(glload)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glload/include")


  # Build vislib
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
 
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
  
  include_directories("include" "src")

  source_group("Header Files" FILES "include/*.h src/*.h")
  source_group("Source Files" FILES "src/*.cpp")
  
  # Target definition
  add_library(vislib STATIC ${header_files} ${source_files})
  add_dependencies(vislib glload)
  target_link_libraries(vislib PRIVATE ${LIBS})
  

  # build tests if variable VISLIB_BUILD_TESTS is defined
  option(BUILD_VISLIB_TESTS "Build vislib tests" OFF)
  mark_as_advanced(BUILD_VISLIB_TESTS)
  if (VISLIB_BUILD_TESTS)
	  set(VISLIB_BUILD_TESTS ${VISLIB_BUILD_TESTS})
	  add_subdirectory(tests)
  endif()


  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(TARGETS vislib DESTINATION "lib" EXPORT vislib-target)

  
endif(BUILD_VISLIB)
