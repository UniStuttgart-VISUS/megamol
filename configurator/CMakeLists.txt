#
# MegaMol™ Configurator
# Copyright 2017, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#

option(BUILD_CONFIGURATOR "Build configurator" ON)

set(PROJ_NAME "MegaMol")
set(PROJ_TEAM "MegaMol Team: Visualization Research Center, University of Stuttgart; TU Dresden")
set(PROJ_DESC "MegaMol Project File Configuration Utility. http://megamol.org")

if(BUILD_CONFIGURATOR)
    project(configurator)
    set(CONFIGURATOR_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/MegaMolConf)
    configure_file(${CONFIGURATOR_SOURCE}/Resources/mmconfig.rc.input ${CONFIGURATOR_SOURCE}/Resources/mmconfig.rc)
endif()