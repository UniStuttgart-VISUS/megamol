/*
 * mmvtkmDataCall.cpp (MultiParticleDataCall)
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmvtkm/mmvtkmDataCall.h"
//#include "vislib/memutils.h"

using namespace megamol;
using namespace megamol::mmvtkm;


/*
 * moldyn::mmvtkmDataCall::mmvtkmDataCall
 */
mmvtkmDataCall::mmvtkmDataCall(void) : AbstractGetData3DCall(), 
	mData(nullptr) 
{
    // Intentionally empty
}


/*
 * moldyn::vtkmDataCall::~vtkmDataCall
 */
mmvtkmDataCall::~mmvtkmDataCall(void) {
    // Intentionally empty
}


/*
 * moldyn::vtkmDataCall::operator=
 */
mmvtkmDataCall& mmvtkmDataCall::operator=(
        const mmvtkmDataCall& rhs) {
    AbstractGetData3DCall::operator=(rhs);
    return *this;
}
