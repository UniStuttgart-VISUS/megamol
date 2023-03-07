/*
 * EllipsoidalDataCall.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team.
 * Alle Rechte vorbehalten.
 */

#include "geometry_calls/EllipsoidalDataCall.h"

namespace megamol::geocalls {

/****************************************************************************/

/*
 * EllipsoidalParticles
 */
EllipsoidalParticles::EllipsoidalParticles()
        : SimpleSphericalParticles()
        , quatPtr(NULL)
        , quatStride(0)
        , radPtr(NULL)
        , radStride(0) {
    // intentionally empty
}


/*
 * EllipsoidalParticles
 */
EllipsoidalParticles::EllipsoidalParticles(const EllipsoidalParticles& src) {
    *this = src;
}


/*
 * ~EllipsoidalParticles
 */
EllipsoidalParticles::~EllipsoidalParticles() {
    this->quatPtr = NULL; // DO NOT DELETE
    this->radPtr = NULL;  // DO NOT DELETE
}


/*
 * operator=
 */
EllipsoidalParticles& EllipsoidalParticles::operator=(const EllipsoidalParticles& rhs) {
    SimpleSphericalParticles::operator=(rhs);
    this->quatPtr = rhs.quatPtr;
    this->quatStride = rhs.quatStride;
    this->radPtr = rhs.radPtr;
    this->radStride = rhs.radStride;
    return *this;
}


/*
 * operator==
 */
bool EllipsoidalParticles::operator==(const EllipsoidalParticles& rhs) const {
    return SimpleSphericalParticles::operator==(rhs) && (this->quatPtr == rhs.quatPtr) &&
           (this->quatStride == rhs.quatStride) && (this->radPtr == rhs.radPtr) && (this->radStride == rhs.radStride);
}

/****************************************************************************/

const unsigned int EllipsoidalParticleDataCall::CallForGetData = 0;
const unsigned int EllipsoidalParticleDataCall::CallForGetExtents = 1;

/*
 * EllipsoidalParticleDataCall
 */
EllipsoidalParticleDataCall::EllipsoidalParticleDataCall() : AbstractParticleDataCall<EllipsoidalParticles>() {
    // Intentionally empty
}


/*
 * ~EllipsoidalParticleDataCall
 */
EllipsoidalParticleDataCall::~EllipsoidalParticleDataCall() {
    // Intentionally empty
}


/*
 * operator=
 */
EllipsoidalParticleDataCall& EllipsoidalParticleDataCall::operator=(const EllipsoidalParticleDataCall& rhs) {
    AbstractParticleDataCall<EllipsoidalParticles>::operator=(rhs);
    return *this;
}

} // namespace megamol::geocalls
