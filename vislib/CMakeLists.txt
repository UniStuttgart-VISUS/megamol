option(BUILD_VISLIB "Build vislib" ON)

if(BUILD_VISLIB)
  project(vislib)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  require_external(spdlog)
  require_external(libpng)

  # Build vislib
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

  # Target definition
  add_library(vislib STATIC ${public_header_files} ${header_files} ${source_files})

  target_link_libraries(vislib 
    PRIVATE ${CMAKE_THREAD_LIBS_INIT} libpng
    PUBLIC spdlog)

  # Word size
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BITS 64)
  else()
    set(BITS 32)
  endif()

  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src")
  target_compile_definitions(vislib
    PUBLIC WIN32_LEAN_AND_MEAN "_LIN${BITS}"
    PRIVATE _GNU_SOURCE)

  if(UNIX)
    find_package(UUID REQUIRED)
    set(CURSES_NEED_NCURSES True)
    find_package(Curses REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${UUID_LIBRARIES} ${CURSES_LIBRARIES})
  endif()

  # Grouping in Visual Studio
  set_target_properties(vislib PROPERTIES FOLDER base)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

  # Installation rules for generated files
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  #install(TARGETS vislib DESTINATION "lib" EXPORT vislib-target)
endif()
