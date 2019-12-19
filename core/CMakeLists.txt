#
# MegaMol™ Core
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
option(BUILD_CORE "Build core" ON)

if(BUILD_CORE)
  project(core)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_CORE BUILD_VISLIB" CACHE INTERNAL "")

  # Add expat.
  if(WIN32)
    option(USE_INTERNAL_EXPAT "Compiles the expat provided by MegaMol" ON)
    if(USE_INTERNAL_EXPAT)
      add_subdirectory(expat)
      set(EXPAT_LIBRARIES expat)
      set(EXPAT_INCLUDE_DIRS "")
    else()
      find_package(EXPAT REQUIRED)
    endif()
  else()
    find_package(EXPAT REQUIRED)
  endif()

  # Add Lua.
  add_subdirectory(lua)

  # Add externals.
  require_external(libzmq)
  require_external(libcppzmq)
  require_external(zlib)
  require_external(libpng)
  require_external(glm)
  require_external(glowl)
  require_external(json)

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h" "include/*.inl")
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
    PUBLIC ${EXPAT_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} 
    PUBLIC vislib glad glm glowl zlib libpng json lua ${EXPAT_LIBRARIES}
    PRIVATE libzmq libcppzmq ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})

  # Add libunwind.
  if(UNIX)
    option(USE_LIBUNWIND "use libunwind" OFF)
    if(USE_LIBUNWIND)
      find_package(LIBUNWIND REQUIRED)
      message(STATUS ${LIBUNWIND_LIBRARIES})
      message(STATUS ${LIBUNWIND_INCLUDE_DIR})
      if(LIBUNWIND_FOUND)
        target_compile_definitions(${PROJECT_NAME} PUBLIC "HAS_LIBUNWIND")
        target_include_directories(${PROJECT_NAME} PUBLIC ${LIBUNWIND_INCLUDE_DIR})
        target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBUNWIND_LIBRARIES} )
      else()
        message(FATAL_ERROR "Can not find libunwind.")
      endif()
    endif()
  endif()
  
  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
      LINK_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DYNAMICBASE:NO")
  endif()

  if(MPI_C_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_C)
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
  get_target_property(zmqcpp_include libcppzmq INTERFACE_INCLUDE_DIRECTORIES)
  install(DIRECTORY ${zmqcpp_include} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/libzmq FILES_MATCHING PATTERN "*.hpp")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    if (MSVC)
      install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
    endif()
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  add_subdirectory(remoteconsole)
endif(BUILD_CORE)
