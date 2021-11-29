/*
 * QuartzParticleGridDataCall.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "QuartzParticleGridDataCall.h"
#include "stdafx.h"

namespace megamol {
namespace demos_gl {


/*
 * ParticleGridDataCall::CallForGetData
 */
const unsigned int ParticleGridDataCall::CallForGetData = 0;


/*
 * ParticleGridDataCall::CallForGetExtent
 */
const unsigned int ParticleGridDataCall::CallForGetExtent = 1;


/*
 * ParticleGridDataCall::ParticleGridDataCall
 */
ParticleGridDataCall::ParticleGridDataCall(void) : core::AbstractGetData3DCall(), cells(NULL), sx(0), sy(0), sz(0) {
    // intentionally empty
}


/*
 * ParticleGridDataCall::~ParticleGridDataCall
 */
ParticleGridDataCall::~ParticleGridDataCall(void) {
    this->sx = this->sy = this->sz = 0;
    this->cells = NULL; // DO NOT DELETE
}

} // namespace demos_gl
} /* end namespace megamol */
