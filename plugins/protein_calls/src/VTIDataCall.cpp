//
// VTIDataCall.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 16, 2013
//     Author: scharnkn
//

#include "protein_calls/VTIDataCall.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::protein_calls;

const unsigned int VTIDataCall::CallForGetData = 0;
const unsigned int VTIDataCall::CallForGetExtent = 1;

/*
 * VTIDataCall::VTIDataCall
 */
VTIDataCall::VTIDataCall(void) : core::AbstractGetData3DCall(), calltime(0.0) {
    // intentionally empty
}

/*
 * VTIDataCall::~VTIDataCall
 */
VTIDataCall::~VTIDataCall(void) {}
