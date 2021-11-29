/*
 * ParticleDataCall.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "protein/ParticleDataCall.h"
#include "stdafx.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/math/mathfunctions.h"

using namespace megamol;
using namespace megamol::protein;


/*
 * ParticleDataCall::CallForGetData
 */
const unsigned int protein::ParticleDataCall::CallForGetData = 0;

/*
 * ParticleDataCall::CallForGetExtent
 */
const unsigned int protein::ParticleDataCall::CallForGetExtent = 1;

/*
 * protein::ParticleDataCall::ParticleDataCall
 */
protein::ParticleDataCall::ParticleDataCall(void) : AbstractGetData3DCall(), particleCount(0), particles(0) {
    // intentionally empty
}


/*
 * protein::ParticleDataCall::~ParticleDataCall
 */
protein::ParticleDataCall::~ParticleDataCall(void) {}
