#
# MegaMol™ adios_plugin Plugin
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER adios_plugin EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build adios_plugin" OFF)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  project(adios_plugin)

  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_MMSTD_DATATOOLS_PLUGIN" CACHE INTERNAL "")

  option(USE_EXTERNAL_ADIOS "Option to externally include ADIOS2" OFF)
  if (USE_EXTERNAL_ADIOS)
    find_package(ADIOS2 REQUIRED)
  else()
    add_external_project(adios2_ext
    GIT_REPOSITORY https://github.com/ornladios/ADIOS2.git
    GIT_TAG "v2.3.1"
    CMAKE_ARGS -DBUILD_TESTING=OFF -DADIOS2_USE_BZip2=OFF 
    -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_HDF5=OFF 
    -DADIOS2_USE_Python=OFF -DADIOS2_USE_SST=OFF 
    -DADIOS2_USE_SZ=OFF -DADIOS2_USE_SysVShMem=OFF 
    -DADIOS2_USE_ZFP=OFF -DADIOS2_USE_ZeroMQ=OFF 
    -DMPI_GUESS_LIBRARY_NAME=${MPI_GUESS_LIBRARY_NAME})

    if(WIN32)
      set(ADIOS2_IMPORT_LIB "lib/adios2.lib")
      set(ADIOS2_IMPORT_LIB_DEBUG ${ADIOS2_IMPORT_LIB})
      set(ADIOS2_LIB "bin/adios2.dll")
      set(ADIOS2_LIB_DEBUG ${ADIOS2_LIB})
    else()
      include(GNUInstallDirs)
      set(ADIOS2_IMPORT_LIB_DEBUG "lib/libadios2.so")
      set(ADIOS2_IMPORT_LIB "lib/libadios2.so")
      set(ADIOS2_LIB_DEBUG "lib/libadios2.so")
      set(ADIOS2_LIB "lib/libadios2.so")
    endif()
    add_external_library(adios2 SHARED
    DEPENDS adios2_ext
    IMPORT_LIBRARY_DEBUG ${ADIOS2_IMPORT_LIB_DEBUG}
    IMPORT_LIBRARY_RELEASE ${ADIOS2_IMPORT_LIB}
    LIBRARY_DEBUG ${ADIOS2_LIB_DEBUG}
    LIBRARY_RELEASE ${ADIOS2_LIB})
  endif()

  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  #file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")

  # Target definition
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files})
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
    # Set target naming conventions for different build types
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  if(USE_EXTERNAL_ADIOS)
    target_link_libraries(${PROJECT_NAME} PRIVATE core mmstd_datatools adios2::adios2)
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE core mmstd_datatools adios2)
  endif()
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> "include" "src")

  if(MPI_C_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_C_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})
  endif()

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")

  # if(UNIX)
  #   include(GNUInstallDirs)
  #   get_target_property(adios_lib adios2::adios2 IMPORTED_LOCATION_RELEASE)
  #   get_filename_component(dir ${adios_lib} DIRECTORY)
  #   file(GLOB ADIOS_LIB_FILES "${dir}/lib*")
  #   install(FILES ${ADIOS_LIB_FILES} DESTINATION ${EXTERNAL_INSTALL_LOCATION}/${CMAKE_INSTALL_LIBDIR})
  # endif()

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  if(NOT USE_EXTERNAL_ADIOS)
    install_external(TARGETS adios2)
  endif()
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
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
    string(REGEX REPLACE "^include\\\\adios_plugin" "Public Header Files" GROUP_NAME ${GROUP_NAME})
    source_group(${GROUP_NAME} FILES ${FILE_NAME})
  endforeach()

  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
endif(BUILD_${EXPORT_NAME}_PLUGIN)
