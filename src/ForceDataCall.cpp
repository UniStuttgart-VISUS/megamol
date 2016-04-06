/*
 * ForceDataCall.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "ForceDataCall.h"

using namespace megamol;
using namespace protein_cuda;

/*
 * ForceDataCall::CallForGetForceData
 */
const unsigned int protein_cuda::ForceDataCall::CallForGetForceData = 0;


/*
 * ForceDataCall::ForceDataCall
 */
ForceDataCall::ForceDataCall(void) : forceCount(0), forceAtomIDs(NULL), forceArray(NULL) { 
    // intentionally empty
}

/*
 * ForceDataCall::~ForceDataCall
 */
ForceDataCall::~ForceDataCall(void) {
    // intentionally empty
}

/*
 * ForceDataCall::SetForces
 */
void ForceDataCall::SetForces(unsigned int count, const unsigned int *atomIDs, const float *forces){
    this->forceCount = count;
    this->forceAtomIDs = atomIDs;
    this->forceArray = forces;
}