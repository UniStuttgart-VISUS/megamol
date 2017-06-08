/*
 * CallVolumeData.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/CallVolumeData.h"


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


/*
 * megamol::core::CallVolumeData::operator=
 */
megamol::core::CallVolumeData& megamol::core::CallVolumeData::operator=(const megamol::core::CallVolumeData& rhs) {
    AbstractGetData3DCall::operator=(rhs);
    this->xSize = rhs.xSize;
    this->ySize = rhs.ySize;
    this->zSize = rhs.zSize;
    this->attributes = rhs.attributes;

    return *this;
}
