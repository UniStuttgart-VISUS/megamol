/*
 * CallVolumeData.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallVolumeData.h"


/*
 * megamol::core::CallVolumeData::CallVolumeData
 */
megamol::core::CallVolumeData::CallVolumeData(void) : xSize(0), ySize(0), zSize(0), attributes(0, 1) {
    // intentionally empty
}


/*
 * megamol::core::CallVolumeData::~CallVolumeData
 */
megamol::core::CallVolumeData::~CallVolumeData(void) {
    this->attributes.Clear(true); // paranoia
}
