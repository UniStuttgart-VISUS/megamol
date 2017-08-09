#
# MegaMol™ Console Front End
# Copyright 2015, by MegaMol TEAM
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_CONSOLE "build console" ON)
if(BUILD_CONSOLE)

  project(${CONSOLE_NAME})
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  if(UNIX)
    option(USE_LIBUNWIND "use libunwind" ON)
    if(USE_LIBUNWIND)
      find_package(LIBUNWIND REQUIRED)
      message(STATUS ${LIBUNWIND_LIBRARIES})
      message(STATUS ${LIBUNWIND_INCLUDE_DIR})
      if (LIBUNWIND_FOUND)
	      add_definitions("-DHAS_LIBUNWIND")
      else()
	      message(FATAL_ERROR "Cannot find Libunwind")
      endif()
    endif(USE_LIBUNWIND)
    
    if(USE_EGL)
        find_package(EGL REQUIRED)
    endif()
  endif()

  if(WIN32)
    set(VERSION_LIBRARY "Version.lib")
  endif()
  
  if(NOT USE_EGL)
    option(USE_INTERNAL_GLFW "use internal GLFW" ON)
    if(USE_INTERNAL_GLFW)
        add_subdirectory(glfw)
        set(GLFW3_LIBRARIES glfw)
    else()
        find_package(glfw3 3.2 MODULE)
        if(NOT GLFW3_FOUND)
        find_package(glfw3 3.2 CONFIG REQUIRED)
        set(GLFW3_LIBRARIES glfw)
        # include dirs are included in the imported library (add_library(glfw STATIC IMPORTED))
        set(GLFW3_INCLUDE_DIRS "")
        endif()
    endif()
  endif()

  set(LIBS vislib core ${GLFW3_LIBRARIES} ${VERSION_LIBRARY} ${LIBUNWIND_LIBRARIES} ${EGL_LIBRARIES})
  include_directories(${GLFW3_INCLUDE_DIRS} ${LIBUNWIND_INCLUDE_DIR} ${EGL_INCLUDE_DIR})

  if(NOT USE_EGL)
    option(USE_INTERNAL_ANTTWEAKBAR "use internal AntTweakBar" ON)
    if(USE_INTERNAL_ANTTWEAKBAR)
        add_subdirectory(AntTweakBar)
        add_definitions("-DHAS_ANTTWEAKBAR")
        list(APPEND LIBS ${AntTweakBar_LIBRARIES})
        include_directories(${AntTweakBar_INCLUDE_DIR})
    else()
        find_package(AntTweakBar REQUIRED)
        if (AntTweakBar_FOUND)
        add_definitions("-DHAS_ANTTWEAKBAR")
        list(APPEND LIBS ${AntTweakBar_LIBRARIES})
        include_directories(${AntTweakBar_INCLUDE_DIR})
        endif()
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
  
  # main target
  add_executable(${PROJECT_NAME} ${header_files} ${source_files})
  add_dependencies(${PROJECT_NAME} vislib core)
  # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} 
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME ${PROJECT_NAME})
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

  set_target_properties(${CONSOLE_NAME} PROPERTIES FOLDER base)
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})

  # install target
  install(TARGETS ${PROJECT_NAME} DESTINATION "bin")
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

  set(cfg_MEGAMOLCON "${CMAKE_INSTALL_PREFIX}/bin/${PROJECT_NAME}")

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
