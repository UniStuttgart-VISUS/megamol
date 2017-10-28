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

  if (MSVC_IDE)
    set (MSVC_TOOLSET "-${CMAKE_VS_PLATFORM_TOOLSET}")
  else ()
    set (MSVC_TOOLSET "")
  endif ()

  # include ZMQ
  include(ExternalProject)

  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_SOURCE_DIR}/external CACHE INTERNAL '')
  
  set(ZMQ_VER "4_2_2")
  string(REPLACE "_" "." ZMQ_TAG ${ZMQ_VER})
  string(CONCAT ZMQ_TAG "v" ${ZMQ_TAG})
  if (MSVC)
    set(ZMQ_LIB debug "${EXTERNAL_INSTALL_LOCATION}/lib/libzmq${MSVC_TOOLSET}-mt-gd-${ZMQ_VER}.lib" optimized "${EXTERNAL_INSTALL_LOCATION}/lib/libzmq${MSVC_TOOLSET}-mt-${ZMQ_VER}.lib" CACHE INTERNAL '')
    set(ZMQ_DLLDIR "${EXTERNAL_INSTALL_LOCATION}/bin")
  else()
    include(GNUInstallDirs)
    set(ZMQ_LIB "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libzmq.so")
  endif()
  ExternalProject_Add(extlibzmq
    GIT_REPOSITORY https://github.com/zeromq/libzmq.git
    GIT_TAG ${ZMQ_TAG}
    CMAKE_ARGS -DZMQ_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DCMAKE_BUILD_TYPE=release
  )
  message(STATUS "ZMQ Git Tag: " ${ZMQ_TAG})

  # very ugly way of pulling cppzmq (also caused by the weird visual studio lib naming scheme of zmq)
  if(UNIX)
    set(ZMQCPP_INSTALL_CMD "eval")
    set(ZMQCPP_INSTALL_ARGS "cp ${PROJECT_BINARY_DIR}/extlibcppzmq-prefix/src/extlibcppzmq/*.hpp ${EXTERNAL_INSTALL_LOCATION}/include/")
  else()
    set(ZMQCPP_INSTALL_CMD "powershell")
    set(ZMQCPP_INSTALL_ARGS "copy ${PROJECT_BINARY_DIR}/extlibcppzmq-prefix/src/extlibcppzmq/*.hpp ${EXTERNAL_INSTALL_LOCATION}/include/")
  endif()
  
  ExternalProject_Add(extlibcppzmq
    DEPENDS extlibzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG ${ZMQ_TAG}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${ZMQCPP_INSTALL_CMD} ${ZMQCPP_INSTALL_ARGS}
  )

  
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

  if(UNIX)
    find_package(OpenMP)
  endif()

  find_package(MPI)

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
  set(ZFP_VER "0_5_2")
  string(REPLACE "_" "." ZFP_TAG ${ZFP_VER})
  if (MSVC)
    set(ZFP_LIB "${EXTERNAL_INSTALL_LOCATION}/lib/zfp.lib" CACHE INTERNAL '')
  else()
    include(GNUInstallDirs)
    set(ZFP_LIB "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libzfp.a")
  endif()
  ExternalProject_Add(extzfp
    GIT_REPOSITORY https://github.com/LLNL/zfp.git
    GIT_TAG ${ZFP_TAG}
    CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF -DBUILD_UTILITIES=OFF -DBUILD_TESTING=OFF -DZFP_WITH_ALIGNED_ALLOC=ON -DZFP_WITH_CACHE_FAST_HASH=ON -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DCMAKE_BUILD_TYPE=release
  )
  message(STATUS "ZFP Git Tag: " ${ZFP_TAG})

  set(LIBS vislib glad lua ${EXPAT_LIBRARIES} ${ZLIB_LIBRARIES} ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${ZMQ_LIB})
  include_directories(${EXPAT_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS} ${EXTERNAL_INSTALL_LOCATION}/include "src" "include")
  link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

  if (OPENMP_FOUND OR WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
  if (MPI_C_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_C_COMPILE_FLAGS}")
    include_directories(${MPI_C_INCLUDE_PATH})
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}")
    set(LIBS ${LIBS} ${MPI_C_LIBRARIES})
    add_definitions(-DWITH_MPI -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
  endif()


  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")
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
  add_library(core SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_dependencies(core vislib glad extlibzmq extlibcppzmq)
  # Note: linking private prevents that dependent projects are required to link agains all these again.
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(core PRIVATE ${LIBS})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_compile_definitions(${PROJECT_NAME} PRIVATE MEGAMOLCORE_EXPORTS)
  if(WIN32)
    set_target_properties(core PROPERTIES LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DYNAMICBASE:NO")
  endif()

  # Target structure in Visual Studio
  set_target_properties(core PROPERTIES FOLDER base)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Shaders" FILES ${shader_files})

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    install(TARGETS core RUNTIME DESTINATION "bin")
    install(DIRECTORY ${ZMQ_DLLDIR} DESTINATION ${CMAKE_INSTALL_PREFIX} FILES_MATCHING PATTERN "libzmq*.dll")
    install(TARGETS core ARCHIVE DESTINATION "lib")
  else()
    file(GLOB ZMQ_LIB_FILES "${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libzmq.so*")
    install(FILES ${ZMQ_LIB_FILES} DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
    install(TARGETS core DESTINATION "lib" EXPORT core-target)
  endif()

add_subdirectory(remoteconsole)

endif(BUILD_CORE)

