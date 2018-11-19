#
# MegaMol™ Console Front End
# Copyright 2017, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_CONSOLE "build console" ON)

if(BUILD_CONSOLE)
  project(${CONSOLE_NAME})
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_MMSTD_DATATOOLS_PLUGIN BUILD_CORE" CACHE INTERNAL "")

  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")

  # Add target
  add_executable(${PROJECT_NAME} ${header_files} ${source_files})
  set_target_properties(${PROJECT_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME ${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME} PRIVATE
    ${LIBUNWIND_INCLUDE_DIR} "src" "include")
  target_link_libraries(${PROJECT_NAME} PRIVATE core)

  if(UNIX)
    option(USE_LIBUNWIND "use libunwind" ON)
    if(USE_LIBUNWIND)
      find_package(LIBUNWIND REQUIRED)
      message(STATUS ${LIBUNWIND_LIBRARIES})
      message(STATUS ${LIBUNWIND_INCLUDE_DIR})
      if(LIBUNWIND_FOUND)
        target_compile_definitions(${PROJECT_NAME} PRIVATE "HAS_LIBUNWIND=1")
      else()
        message(FATAL_ERROR "Cannot find libunwind")
      endif()
    endif()
    if(USE_EGL)
      find_package(EGL REQUIRED)
      target_include_directories(${PROJECT_NAME} PRIVATE ${EGL_INCLUDE_DIR})
      target_link_libraries(${PROJECT_NAME} PRIVATE ${EGL_LIBRARIES})
    endif()
  endif()

  if(WIN32)
    set(VERSION_LIBRARY "Version.lib")
  endif()

  if(NOT USE_EGL)  
    bootstrap_external_package(glfw_ext glfw3
      CONFIG "lib/cmake/glfw3/glfw3Config.cmake"
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG "3.2.1"
      CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DGLFW_BUILD_EXAMPLES=OFF
        -DGLFW_BUILD_TESTS=OFF
        -DGLFW_BUILD_DOCS=OFF)
    find_package(glfw3 REQUIRED CONFIG)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
  endif()
  
  # AntTweakBar
  if(NOT USE_EGL AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    option(USE_INTERNAL_ANTTWEAKBAR "use internal AntTweakBar" ON)
    if(USE_INTERNAL_ANTTWEAKBAR)
      add_subdirectory(AntTweakBar)
    else()
      find_package(AntTweakBar REQUIRED)
    endif()
    if(AntTweakBar_FOUND OR USE_INTERNAL_ANTTWEAKBAR)
      target_compile_definitions(${PROJECT_NAME} PRIVATE "HAS_ANTTWEAKBAR=1")
      target_include_directories(${PROJECT_NAME} PRIVATE ${AntTweakBar_INCLUDE_DIR})
      target_link_libraries(${PROJECT_NAME} PRIVATE ${AntTweakBar_LIBRARIES})
    endif()
  endif()
  
  target_link_libraries(${PROJECT_NAME} PRIVATE
	  ${VERSION_LIBRARY} ${LIBUNWIND_LIBRARIES} ${CMAKE_DL_LIBS})

  # Grouping in Visual Studio
  set_target_properties(${CONSOLE_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

  # install and configure ATB
  install(TARGETS ${PROJECT_NAME} DESTINATION "bin")
  if(AntTweakBar_FOUND)
    install(FILES ${AntTweakBar_LIBRARIES} DESTINATION "lib")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/linkAntTweakBar.cmake.in ${CMAKE_BINARY_DIR}/linkAntTweakBar.cmake @ONLY)
    install(SCRIPT ${CMAKE_BINARY_DIR}/linkAntTweakBar.cmake)
    if(WIN32)
      install(FILES ${AntTweakBar_DLL} DESTINATION "bin")
    endif()
  endif()

  # Configuration of the megamol.cfg megamol.sh remoteconsole.sh files
  set(cfg_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(cfg_MEGAMOLCON "${CMAKE_INSTALL_PREFIX}/bin/${PROJECT_NAME}")

  if(UNIX)
    include(GNUInstallDirs)
    set(cfg_EXTERNAL_LIB_PATH "${CMAKE_INSTALL_LIBDIR}")
    set(MEGAMOL_SHELL_START_SCRIPT "megamol.sh")
    configure_file(extra/megamol.sh.in ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} @ONLY)
    install(PROGRAMS ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} DESTINATION "bin")
  endif()

  set(MMPLG_DIR "lib")
  if(WIN32)
    set(MMPLG_DIR "bin")
  endif()

  configure_file(extra/megamolconfig.lua.in ${CMAKE_BINARY_DIR}/megamolconfig.lua)
  configure_file(extra/install_megamolconfig_lua.cmake.in ${CMAKE_BINARY_DIR}/install_megamolconfig_lua.cmake @ONLY)
  install(SCRIPT ${CMAKE_BINARY_DIR}/install_megamolconfig_lua.cmake)
endif()
