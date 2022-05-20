/*
 * MultiParticleDataCall.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "geometry_calls/MultiParticleDataCall.h"
//#include "vislib/memutils.h"

namespace megamol::geocalls {


/*
 * MultiParticleDataCall::MultiParticleDataCall
 */
MultiParticleDataCall::MultiParticleDataCall(void) : AbstractParticleDataCall<SimpleSphericalParticles>() {
    // Intentionally empty
}


/*
 * MultiParticleDataCall::~MultiParticleDataCall
 */
MultiParticleDataCall::~MultiParticleDataCall(void) {
    // Intentionally empty
}


/*
 * MultiParticleDataCall::operator=
 */
MultiParticleDataCall& MultiParticleDataCall::operator=(const MultiParticleDataCall& rhs) {
    AbstractParticleDataCall<SimpleSphericalParticles>::operator=(rhs);
    return *this;
}
} // namespace megamol::geocalls
