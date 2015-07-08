/*
 * VolumeSliceCall.cpp
 *
 * Author: Michael Krone
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "VolumeSliceCall.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;
using namespace megamol::protein;

/*
 * MolecularDataCall::CallForGetData
 */
const unsigned int protein::VolumeSliceCall::CallForGetData = 0;

/*
 * VolumeSliceCall::VolumeSliceCall
 */
VolumeSliceCall::VolumeSliceCall(void) : Call(),
        volumeTex( 0), texRCoord( 0.5f), clipPlaneNormal( 0,-1, 0),
        bBoxDim( 1, 1, 1), mousePos( 0, 0, 0), clickPos( 0, 0, 0) {
    // intentionally empty
}


/*
 * VolumeSliceCall::~VolumeSliceCall
 */
VolumeSliceCall::~VolumeSliceCall(void) {

}
