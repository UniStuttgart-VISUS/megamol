#
# MegaMol™ Core
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

option(BUILD_CORE "Build core" ON)

if(BUILD_CORE)
  project(core)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_CORE BUILD_VISLIB" CACHE INTERNAL "")

  # search for 3rd party libs
  set(CMAKE_THREAD_PREFER_PTHREAD)

  find_package(Threads REQUIRED)

  # include lua sources
  add_subdirectory(lua)

  # include ZMQ and its C++ bindings
  set(ZMQ_VER "4_2_3")
  string(REPLACE "_" "." ZMQ_TAG "v${ZMQ_VER}")
  if(MSVC_IDE)
    set(MSVC_TOOLSET "-${CMAKE_VS_PLATFORM_TOOLSET}")
  else()
    set(MSVC_TOOLSET "")
  endif()
  if(WIN32)
    set(ZMQ_IMPORT_DEBUG "lib/libzmq${MSVC_TOOLSET}-mt-gd-${ZMQ_VER}.lib")
    set(ZMQ_IMPORT_RELEASE "lib/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.lib")
    set(ZMQ_DEBUG "bin/libzmq${MSVC_TOOLSET}-mt-gd-${ZMQ_VER}.dll")
    set(ZMQ_RELEASE "bin/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.dll")
  else()
    include(GNUInstallDirs)
    set(ZMQ_IMPORT_DEBUG "")
    set(ZMQ_IMPORT_RELEASE "")
    set(ZMQ_DEBUG "${CMAKE_INSTALL_LIBDIR}/libzmq.so")
    set(ZMQ_RELEASE ${ZMQ_DEBUG})
  endif()

  add_external_project(libzmq_ext
    GIT_REPOSITORY https://github.com/zeromq/libzmq.git
    GIT_TAG ${ZMQ_TAG}
    CMAKE_ARGS
      -DZMQ_BUILD_TESTS=OFF)
  add_external_library(libzmq SHARED
    DEPENDS libzmq_ext
    IMPORT_LIBRARY_DEBUG ${ZMQ_IMPORT_DEBUG}
    IMPORT_LIBRARY_RELEASE ${ZMQ_IMPORT_RELEASE}
    LIBRARY_DEBUG ${ZMQ_DEBUG}
    LIBRARY_RELEASE ${ZMQ_RELEASE})

  add_external_project(libcppzmq_ext
    DEPENDS libzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG ${ZMQ_TAG}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND "")
  add_external_library(libcppzmq INTERFACE
    DEPENDS libcppzmq_ext
    INCLUDE_DIR "src/libcppzmq_ext/")

  option(USE_INTERNAL_LIBPNG "adds libpng to the build targets" ON)
  if(USE_INTERNAL_LIBPNG)
    add_subdirectory(zlib)
    add_subdirectory(libpng)
  else()
    find_package(ZLIB MODULE)
    find_package(PNG MODULE)
    if(NOT PNG_FOUND)
      find_package(libpng16 CONFIG REQUIRED CONFIGS "libpng16.cmake")
      get_target_property(PNG_LIB_FILE png_static LOCATION)
      get_filename_component(PNG_LIB_PATH ${PNG_LIB_FILE} DIRECTORY)
      set(PNG_LIBRARIES png_static CACHE INTERNAL "")
      set(PNG_INCLUDE_DIRS "${PNG_LIB_PATH}/../include" CACHE INTERNAL "")
    endif()
  endif()

  if(WIN32)
    option(USE_INTERNAL_EXPAT "Compiles the expat provided by megamol" ON)
    if(USE_INTERNAL_EXPAT)
      add_subdirectory(expat)
      set(EXPAT_LIBRARIES expat)
    else()
      find_package(EXPAT REQUIRED)
    endif()
  else()
    find_package(EXPAT REQUIRED)
  endif()

  # include zfp
  if(WIN32)
    set(ZFP_LIB "zfp_ext-build/lib/zfp.lib")
  else()
    include(GNUInstallDirs)
    set(ZFP_LIB "zfp_ext-build/${CMAKE_INSTALL_LIBDIR}/libzfp.a")
  endif()

  add_external_project(zfp_ext
    GIT_REPOSITORY https://github.com/LLNL/zfp.git
    GIT_TAG "0.5.2"
    CMAKE_ARGS
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_UTILITIES=OFF
      -DBUILD_TESTING=OFF
      -DZFP_WITH_ALIGNED_ALLOC=ON
      -DZFP_WITH_CACHE_FAST_HASH=ON
      -DCMAKE_BUILD_TYPE=Release)
  add_external_library(zfp STATIC
    DEPENDS zfp_ext
    LIBRARY_DEBUG ${ZFP_LIB}
    LIBRARY_RELEASE ${ZFP_LIB})

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")
  file(GLOB_RECURSE resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Resources/*")
  set(version_files ${CMAKE_BINARY_DIR}/version/MMCoreVersionInfo.cpp)

  list(REMOVE_ITEM source_files
    "src/moldyn/D3D11SimpleSphereRenderer.cpp"
    "src/special/AbstractStereoDisplay.cpp"
    "src/special/ColStereoDisplay.cpp"
    "src/utility/AbstractD3D11RenderObject.cpp"
    "src/utility/D3D11BoundingBox.cpp"
    "src/view/CallUpdateDirect3D.cpp"
    "src/view/ViewDirect3D.cpp"
    "src/vismol2/Mol20DataCall.cpp"
    "src/vismol2/Mol20DataSource.cpp"
    "src/vismol2/Mol20Renderer.cpp"
    "src/vismol2/Mol2Data.cpp"
    )

  # Target definition
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files} ${version_files})
  target_compile_definitions(${PROJECT_NAME} PRIVATE MEGAMOLCORE_EXPORTS)
  target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "src" "include"
    PUBLIC ${EXPAT_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE vislib glad lua libzmq libcppzmq ${EXPAT_LIBRARIES} ${ZLIB_LIBRARIES} ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
      LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DYNAMICBASE:NO")
  endif()
  if(MPI_C_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_C_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})
  endif()

  # Source groups in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER base)
  foreach(FILE_NAME ${source_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^src" "Source Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  foreach(FILE_NAME ${header_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^src" "Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  foreach(FILE_NAME ${public_header_files})
    get_filename_component(FILE_PATH ${FILE_NAME} PATH)
    string(REPLACE "/" "\\" GROUP_NAME ${FILE_PATH})
    string(REGEX REPLACE "^include\\\\mmcore" "Public Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()
  source_group("Shaders" FILES ${shader_files})
  source_group("Resources" FILES ${resource_files})
  source_group("Version" FILES ${version_files})

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Resources/ DESTINATION "share/resources")
  install_external(TARGETS libzmq)
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  add_subdirectory(remoteconsole)
endif(BUILD_CORE)
