/*
 * Diagram2DCall.cpp
 *
 * Author: Michael Krone
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "Diagram2DCall.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;
using namespace megamol::protein;

/*
 * MolecularDataCall::CallForGetData
 */
const unsigned int protein::Diagram2DCall::CallForGetData = 0;

/*
 * Diagram2DCall::Diagram2DCall
 */
Diagram2DCall::Diagram2DCall(void) : Call(),
        clearDiagram( false), markerFlag( false) {
    // intentionally empty
}


/*
 * Diagram2DCall::~Diagram2DCall
 */
Diagram2DCall::~Diagram2DCall(void) {

}
