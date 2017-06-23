#
# MegaMol™ Console Front End
# Copyright 2015, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_CONSOLE "build console" ON)
if(BUILD_CONSOLE)

  project(console)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  if(UNIX)
    option(USE_LIBUNWIND "use libunwind" ON)
    if(USE_LIBUNWIND)
      find_package(LIBUNWIND REQUIRED)
      message(STATUS ${LIBUNWIND_LIBRARIES})
      message(STATUS ${LIBUNWIND_INCLUDE_DIR})
      if (LIBUNWIND_FOUND)
	message(STATUS "LIBUNWIND will be used")
	add_definitions("-DHAS_LIBUNWIND")
      else()
	message(FATAL_ERROR "Cannot find Libunwind")
      endif()
    endif(USE_LIBUNWIND)
  endif()

  if(WIN32)
    set(VERSION_LIBRARY "Version.lib")
  endif()
  
  set(LIBS vislib core glload ${VERSION_LIBRARY} ${LIBUNWIND_LIBRARIES})
  include_directories(${VISLIB_INCLUDE_DIRS} ${GLLOAD_INCLUDE_DIRS} ${CORE_INCLUDE_DIRS} ${LIBUNWIND_INCLUDE_DIR})

  find_package(glfw3 3.2 REQUIRED)
  list(APPEND LIBS glfw)

  option(USE_ANTTWEAKBAR "use AntTweakBar" ON)
  if(USE_ANTTWEAKBAR)
    find_package(AntTweakBar REQUIRED)
    if (AntTweakBar_FOUND)
      message(STATUS "AntTweakBar will be used")
      add_definitions("-DHAS_ANTTWEAKBAR")
      list(APPEND LIBS ${AntTweakBar_LIBRARIES})
      include_directories(${AntTweakBar_INCLUDE_DIR})
    endif()
  endif()



  # Build MegaMol Core
  # public header files for installation
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  # source files for compilation
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  #list(REMOVE_ITEM source_files
  #	"src/moldyn/D3D11SimpleSphereRenderer.cpp"
  #	)

  include_directories(BEFORE "src")
  
  source_group("Header Files" FILES "src/*.h")
  source_group("Source Files" FILES "src/*.cpp")

  # main target
  add_executable(console ${header_files} ${source_files})
  # Set target naming conventions for different build types
  set_target_properties(console PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(console PRIVATE ${LIBS})

  # install target
  install(TARGETS console DESTINATION "bin")
  if (AntTweakBar_FOUND)
    install(FILES ${AntTweakBar_LIBRARIES} DESTINATION "lib")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/linkAntTweakBar.cmake.in ${CMAKE_BINARY_DIR}/linkAntTweakBar.cmake @ONLY)
    install(SCRIPT ${CMAKE_BINARY_DIR}/linkAntTweakBar.cmake)
    if(WIN32)
      install(FILES ${AntTweakBar_DLL} DESTINATION "bin")
    endif()
  endif()


  # install target
  set(cfg_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib")

  set(cfg_MEGAMOLCON "${CMAKE_INSTALL_PREFIX}/bin/MegaMolCon")

  if(UNIX)
    set(MEGAMOL_SHELL_START_SCRIPT "megamol.sh")
    configure_file(extra/megamol.sh ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} @ONLY)
    install(PROGRAMS ${CMAKE_BINARY_DIR}/${MEGAMOL_SHELL_START_SCRIPT} DESTINATION "bin")
  endif()

  set(MMPLG_DIR "lib")
  if(WIN32)
    set(MMPLG_DIR "bin")
  endif()


  configure_file(extra/megamol.cfg ${CMAKE_BINARY_DIR}/megamol.cfg)
  configure_file(extra/install_megamol_cfg.cmake.in ${CMAKE_BINARY_DIR}/install_megamol_cfg.cmake @ONLY)
  install(SCRIPT ${CMAKE_BINARY_DIR}/install_megamol_cfg.cmake)

endif(BUILD_CONSOLE)
