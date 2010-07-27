/*
 * SphereDataCall.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "SphereDataCall.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;
using namespace megamol::protein;

/*
 * SphereDataCall::CallForGetData
 */
const unsigned int protein::SphereDataCall::CallForGetData = 0;


/*
 * SphereDataCall::CallForGetExtent
 */
const unsigned int protein::SphereDataCall::CallForGetExtent = 1;

/*
 * protein::SphereDataCall::SphereDataCall
 */
protein::SphereDataCall::SphereDataCall(void) : AbstractGetData3DCall(),
        sphereCount( 0), spheres( 0), colors( 0), charges( 0), 
        minCharge( 0.0f), maxCharge( 0.0f), types( 0) {
    // intentionally empty
}


/*
 * protein::SphereDataCall::~SphereDataCall
 */
protein::SphereDataCall::~SphereDataCall(void) {
}

/*
 * Set the spheres. 
 */
void SphereDataCall::SetSpheres( unsigned int sphereCnt, float* data, 
    unsigned int* type, float* charge, unsigned char* color) {
    // set all values
    this->sphereCount = sphereCnt;
    this->spheres = data;
    this->types = type;
    this->charges = charge;
    this->colors = color;
}
