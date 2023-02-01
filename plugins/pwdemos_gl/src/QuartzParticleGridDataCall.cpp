/*
 * QuartzParticleGridDataCall.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "QuartzParticleGridDataCall.h"

namespace megamol::demos_gl {


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
ParticleGridDataCall::ParticleGridDataCall() : core::AbstractGetData3DCall(), cells(NULL), sx(0), sy(0), sz(0) {
    // intentionally empty
}


/*
 * ParticleGridDataCall::~ParticleGridDataCall
 */
ParticleGridDataCall::~ParticleGridDataCall() {
    this->sx = this->sy = this->sz = 0;
    this->cells = NULL; // DO NOT DELETE
}

} // namespace megamol::demos_gl
