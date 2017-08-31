/*
 * DirectionalParticleDataCall.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/moldyn/EllipsoidalDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::moldyn;

/****************************************************************************/

/*
 * moldyn::DirectionalParticles::DirectionalParticles
 */
EllipsoidalParticles::EllipsoidalParticles(void)
        : SimpleSphericalParticles(), quatPtr(NULL), quatStride(0),
        radPtr(NULL), radStride(0) {
    // intentionally empty
}


/*
 * moldyn::DirectionalParticles::DirectionalParticles
 */
EllipsoidalParticles::EllipsoidalParticles(
    const EllipsoidalParticles& src) {
    *this = src;
}


/*
 * moldyn::DirectionalParticles::~DirectionalParticles
 */
EllipsoidalParticles::~EllipsoidalParticles(void) {
    this->quatPtr = NULL; // DO NOT DELETE
    this->radPtr = NULL; // DO NOT DELETE
}


/*
 * moldyn::DirectionalParticles::operator=
 */
EllipsoidalParticles& EllipsoidalParticles::operator=(
const EllipsoidalParticles& rhs) {
    SimpleSphericalParticles::operator=(rhs);
    this->quatPtr = rhs.quatPtr;
    this->quatStride = rhs.quatStride;
    this->radPtr = rhs.radPtr;
    this->radStride = rhs.radStride;
    return *this;
}


/*
 * moldyn::DirectionalParticles::operator==
 */
bool EllipsoidalParticles::operator==(
    const EllipsoidalParticles& rhs) const {
    return SimpleSphericalParticles::operator==(rhs)
        && (this->quatPtr == rhs.quatPtr)
        && (this->quatStride == rhs.quatStride)
        && (this->radPtr == rhs.radPtr)
        && (this->radStride == rhs.radStride)
        ;
}

/****************************************************************************/

const unsigned int EllipsoidalParticleDataCall::CallForGetData = 0;
const unsigned int EllipsoidalParticleDataCall::CallForGetExtents = 1;

/*
 * moldyn::DirectionalParticleDataCall::DirectionalParticleDataCall
 */
EllipsoidalParticleDataCall::EllipsoidalParticleDataCall(void)
        : AbstractParticleDataCall<EllipsoidalParticles>() {
    // Intentionally empty
}


/*
 * moldyn::DirectionalParticleDataCall::~DirectionalParticleDataCall
 */
EllipsoidalParticleDataCall::~EllipsoidalParticleDataCall(void) {
    // Intentionally empty
}


/*
 * moldyn::DirectionalParticleDataCall::operator=
 */
EllipsoidalParticleDataCall& EllipsoidalParticleDataCall::operator=(
        const EllipsoidalParticleDataCall& rhs) {
    AbstractParticleDataCall<EllipsoidalParticles>::operator =(rhs);
    return *this;
}
