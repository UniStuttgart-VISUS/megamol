/*
 * CallProteinVolumeData.cpp
 *
 * Author: Michael Krone
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "CallProteinVolumeData.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;


/****************************************************************************/

/*
 * protein::CallProteinVolumeData::CallProteinVolumeData
 */
protein::CallProteinVolumeData::CallProteinVolumeData(void) : Call(),
    map( NULL), mapMemory( false), volDim( 0, 0, 0),
    minDensity( 0.0f), maxDensity( 0.0f), meanDensity( 0.0f)
{
    // intentionally empty
}


/*
 * protein::CallProteinVolumeData::~CallProteinVolumeData
 */
protein::CallProteinVolumeData::~CallProteinVolumeData(void) {

}


/*
 * Sets a pointer to the voxel map array.
 */
void protein::CallProteinVolumeData::SetVoxelMapPointer( float *voxelMap) {
    if( this->mapMemory ) delete[] this->map;
    this->map = voxelMap;
    this->mapMemory = false;
}
