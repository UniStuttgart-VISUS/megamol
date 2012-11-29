/*
 * VolumeDataCall.cpp
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "VolumeDataCall.h"

using namespace megamol::core;


/****************************************************************************/

/*
 * VolumeDataCall::CallForGetData
 */
const unsigned int VolumeDataCall::CallForGetData = 0;


/*
 * VolumeDataCall::CallForGetExtent
 */
const unsigned int VolumeDataCall::CallForGetExtent = 1;

/*
 * VolumeDataCall::VolumeDataCall
 */
VolumeDataCall::VolumeDataCall(void) : AbstractGetData3DCall(),
    map( NULL), mapMemory( false), bBox( 0, 0, 0, 0, 0, 0), components( 1), 
    volDim( 0, 0, 0), minDensity( 0.0f), maxDensity( 0.0f), meanDensity( 0.0f)
{
    // intentionally empty
}


/*
 * VolumeDataCall::~VolumeDataCall
 */
VolumeDataCall::~VolumeDataCall(void) {

}


/*
 * Sets a pointer to the voxel map array.
 */
void VolumeDataCall::SetVoxelMapPointer( float *voxelMap) {
    if( this->mapMemory ) delete[] this->map;
    this->map = voxelMap;
    this->mapMemory = false;
}
