/*
 * DataSourceModule.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/DataSourceModule.h"


using namespace megamol::core;


/*
 * view::DataSourceModule::DataSourceModule
 */
view::DataSourceModule::DataSourceModule(void) : Module(), offsetVec(),
        scale(1.0f) {
    // intentionally empty
}


/*
 * view::DataSourceModule::~DataSourceModule
 */
view::DataSourceModule::~DataSourceModule(void) {
    // intentionally empty
}


/*
 * view::DataSourceModule::NumberOfTimeFrames
 */
unsigned int view::DataSourceModule::NumberOfTimeFrames(void) const {
    return 1;
}
