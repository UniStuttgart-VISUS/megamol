/*
 * CallVolumeData.cpp
 *
 * Author: Michael Krone
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "CallVolumeData.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;


/****************************************************************************/

/*
 * protein::CallVolumeData::CallVolumeData
 */
protein::CallVolumeData::CallVolumeData(void) : Call(),
    map( NULL), mapMemory( false), volDim( 0, 0, 0),
    minDensity( 0.0f), maxDensity( 0.0f), meanDensity( 0.0f)
{
    // intentionally empty
}


/*
 * protein::CallVolumeData::~CallVolumeData
 */
protein::CallVolumeData::~CallVolumeData(void) {

}


/*
 * Sets a pointer to the voxel map array.
 */
void protein::CallVolumeData::SetVoxelMapPointer( float *voxelMap) {
    if( this->mapMemory ) delete[] this->map;
    this->map = voxelMap;
    this->mapMemory = false;
}
