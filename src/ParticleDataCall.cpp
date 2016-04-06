/*
 * ParticleDataCall.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "ParticleDataCall.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;
using namespace megamol::protein_cuda;


/*
 * ParticleDataCall::CallForGetData
 */
const unsigned int protein_cuda::ParticleDataCall::CallForGetData = 0;

/*
 * ParticleDataCall::CallForGetExtent
 */
const unsigned int protein_cuda::ParticleDataCall::CallForGetExtent = 1;

/*
 * protein_cuda::ParticleDataCall::ParticleDataCall
 */
protein_cuda::ParticleDataCall::ParticleDataCall(void) : AbstractGetData3DCall(),
        particleCount( 0), particles( 0) {
    // intentionally empty
}


/*
 * protein_cuda::ParticleDataCall::~ParticleDataCall
 */
protein_cuda::ParticleDataCall::~ParticleDataCall(void) {
}
