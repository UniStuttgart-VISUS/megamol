/*
 * MultiParticleDataCall.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
//#include "vislib/memutils.h"

using namespace megamol::core;


/*
 * moldyn::MultiParticleDataCall::MultiParticleDataCall
 */
moldyn::MultiParticleDataCall::MultiParticleDataCall(void)
        : AbstractParticleDataCall<SimpleSphericalParticles>() {
    // Intentionally empty
}


/*
 * moldyn::MultiParticleDataCall::~MultiParticleDataCall
 */
moldyn::MultiParticleDataCall::~MultiParticleDataCall(void) {
    // Intentionally empty
}


/*
 * moldyn::MultiParticleDataCall::operator=
 */
moldyn::MultiParticleDataCall& moldyn::MultiParticleDataCall::operator=(
        const moldyn::MultiParticleDataCall& rhs) {
    AbstractParticleDataCall<SimpleSphericalParticles>::operator =(rhs);
    return *this;
}
