#
# MegaMol™ Configurator
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

option(BUILD_CONFIGURATOR "Build configurator" OFF)

include(CheckLanguage)
check_language(CSharp)
if(BUILD_CONFIGURATOR AND CMAKE_CSharp_COMPILER)
  project(configurator CSharp)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  set(PROJ_NAME "MegaMol")
  set(PROJ_TEAM "MegaMol Team: Visualization Research Center, University of Stuttgart; TU Dresden")
  set(PROJ_DESC "MegaMol Project File Configuration Utility. http://megamol.org")

  string(REGEX REPLACE "/" "\\\\" CONFIGURATOR_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/MegaMolConf")
  string(REGEX REPLACE "/" "\\\\" CONFIGURATOR_BINARY  "${CMAKE_CURRENT_BINARY_DIR}")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MegaMolConf/Resources/mmconfig.rc.input mmconfig.rc)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MegaMolConf/Properties/AssemblyInfo.cs.input AssemblyInfo.cs)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MegaMolConf/MegaMolConf.csproj.input MegaMolConf.csproj)

  include_external_msproject(MegaMolConf ${CMAKE_CURRENT_BINARY_DIR}/MegaMolConf.csproj)

  set_target_properties(MegaMolConf PROPERTIES FOLDER configurator)

  # Copy packages.config to build folder
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/MegaMolConf/packages.config" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  # Run the resource compiler
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/mmconfig.res")
    find_package(RC MODULE)
    if(NOT ${RC_FOUND})
      message(FATAL_ERROR "rc.exe not found")
    endif()

    execute_process(COMMAND "${RC_EXE}" "mmconfig.rc"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    OUTPUT_VARIABLE out_var
                    RESULT_VARIABLE res_var
                    ERROR_VARIABLE err_var)

    if(${res_var} GREATER 0)
      message(FATAL_ERROR "ERROR in ${RC_EXE} mmconfig.rc\nCould not create mmconfig.res\n${out_var}\n${err_var}")
    endif()
  endif()

  # Install binaries and libraries
  install(CODE " \
    file(INSTALL \"${CMAKE_CURRENT_BINARY_DIR}/bin\" \
      DESTINATION ${CMAKE_INSTALL_PREFIX} \
      FILES_MATCHING REGEX \"\\\\.(exe|dll|pdb)$\")")
elseif(BUILD_CONFIGURATOR)
  MESSAGE(WARNING "Cannot build configurator with current generator / OS combination")
endif()
