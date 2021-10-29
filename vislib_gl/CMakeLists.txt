if(BUILD_VISLIB)
  project(vislib_gl)

  require_external(glad)

  # Build vislib
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

  # Target definition
  add_library(vislib_gl STATIC ${public_header_files} ${header_files} ${source_files})

  target_link_libraries(vislib_gl
    PUBLIC glad ${OPENGL_LIBRARIES} vislib)

  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src")

  if(UNIX)
    set(CURSES_NEED_NCURSES True)
    find_package(Curses REQUIRED)
    find_package(UUID REQUIRED)
    find_package(X11 REQUIRED)
    target_link_libraries(vislib_gl PRIVATE ${X11_LIBRARIES} ${UUID_LIBRARIES} ${CURSES_LIBRARIES})
  endif()

  # Grouping in Visual Studio
  set_target_properties(vislib_gl PROPERTIES FOLDER base)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

endif()
