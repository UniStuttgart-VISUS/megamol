/*
 * QuartzParticleDataCall.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "QuartzParticleDataCall.h"


namespace megamol {
namespace demos_gl {


/*
 * ParticleDataCall::CallForGetData
 */
const unsigned int ParticleDataCall::CallForGetData = 0;


/*
 * ParticleDataCall::CallForGetExtent
 */
const unsigned int ParticleDataCall::CallForGetExtent = 1;


/*
 * ParticleDataCall::ParticleDataCall
 */
ParticleDataCall::ParticleDataCall() : core::AbstractGetData3DCall(), grpCnt(0), types(NULL), cnt(NULL), part(NULL) {
    // intentionally empty
}


/*
 * ParticleDataCall::~ParticleDataCall
 */
ParticleDataCall::~ParticleDataCall() {
    this->grpCnt = 0;
    this->types = NULL; // DO NOT DELETE
    this->cnt = NULL;   // DO NOT DELETE
    this->part = NULL;  // DO NOT DELETE
}

} // namespace demos_gl
} /* end namespace megamol */
