#
# MegaMol™ Configurator
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

option(BUILD_CONFIGURATOR "Build configurator" ON)

if(BUILD_CONFIGURATOR)
  
  project(configurator)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  set(PROJ_NAME "MegaMol")
  set(PROJ_TEAM "MegaMol Team: Visualization Research Center, University of Stuttgart; TU Dresden")
  set(PROJ_DESC "MegaMol Project File Configuration Utility. http://megamol.org")

      project(configurator)
      set(CONFIGURATOR_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/MegaMolConf)
      string(REGEX REPLACE "/" "\\\\" CONF_SOURCE  ${CMAKE_CURRENT_SOURCE_DIR})
      string(REGEX REPLACE "/" "\\\\" CONF_BIN  ${CMAKE_CURRENT_BINARY_DIR})
      configure_file(${CONFIGURATOR_SOURCE}/Resources/mmconfig.rc.input mmconfig.rc)
      configure_file(${CONFIGURATOR_SOURCE}/Properties/AssemblyInfo.cs.input AssemblyInfo.cs)

      # Libs and include dirs
  set(LIBS  ${ZMQ_LIB})
  include_directories("MegaMolConf" ${EXTERNAL_INSTALL_LOCATION}/include)

  file(GLOB_RECURSE resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "MegaMolConf/*.resx")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "MegaMolConf/*.cs")

  configure_file(${CONFIGURATOR_SOURCE}/MegaMolConf.csproj.input MegaMolConf.csproj)

  include_external_msproject(conf_csproj "${CMAKE_CURRENT_BINARY_DIR}/MegaMolConf.csproj")

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
endif()