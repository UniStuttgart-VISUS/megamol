#
# MegaMol™ Core
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

option(BUILD_CORE "Build core" ON)

if(BUILD_CORE)
  project(core)

  # search for 3rd party libs
  set(CMAKE_THREAD_PREFER_PTHREAD)

  find_package(Threads REQUIRED)
  
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

  find_package(OpenMP)
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

  set(LIBS vislib glload ${EXPAT_LIBRARIES} ${ZLIB_LIBRARIES} ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  include_directories(${EXPAT_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS})

  if (OPENMP_FOUND)
    #message(STATUS "comp ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
  if (MPI_C_FOUND)
    #message(STATUS "comp ${MPI_C_COMPILE_FLAGS}")
    #message(STATUS "incl ${MPI_C_INCLUDE_PATH}")
    #message(STATUS "link ${MPI_C_LINK_FLAGS}")
    #message(STATUS "libs ${MPI_C_LIBRARIES}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_C_COMPILE_FLAGS}")
    include_directories(${MPI_C_INCLUDE_PATH})
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}")
    set(LIBS ${LIBS} ${MPI_C_LIBRARIES})
    add_definitions(-DWITH_MPI -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
  endif()


  # Build MegaMol Core
  # public header files for installation
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  # source files for compilation
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

  include_directories(BEFORE "include" "src")

  # Target definition
  add_library(core SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_dependencies(core vislib glload)
  # Note: linking private prevents that dependent projects are required to link agains all these again.
  # Dependent projects only need to link against the core.so itself.
  target_link_libraries(core PRIVATE ${LIBS})
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  set_target_properties(core PROPERTIES FOLDER base)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  source_group("Shaders" FILES ${shader_files})

  target_compile_definitions(${PROJECT_NAME} PRIVATE MEGAMOLCORE_EXPORTS)
  if(WIN32)
    set_target_properties(core PROPERTIES LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DYNAMICBASE:NO")
  endif()


  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    install(TARGETS core RUNTIME DESTINATION "bin")
    install(TARGETS core ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS core DESTINATION "lib" EXPORT core-target)
  endif()

endif(BUILD_CORE)

