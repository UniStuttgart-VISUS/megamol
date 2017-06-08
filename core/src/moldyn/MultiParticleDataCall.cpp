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

/****************************************************************************/


/*
 * moldyn::SimpleSphericalParticles::SimpleSphericalParticles
 */
moldyn::SimpleSphericalParticles::SimpleSphericalParticles(void)
        : colDataType(COLDATA_NONE), colPtr(NULL), colStride(0), count(0),
        maxColI(1.0f), minColI(0.0f), radius(0.5f), particleType(0),
        vertDataType(VERTDATA_NONE), vertPtr(NULL), vertStride(0),
		disabledNullChecks(false), 
		clusterInfos(NULL) {
    this->col[0] = 255;
    this->col[1] = 0;
    this->col[2] = 0;
    this->col[3] = 255;
}


/*
 * moldyn::SimpleSphericalParticles::SimpleSphericalParticles
 */
moldyn::SimpleSphericalParticles::SimpleSphericalParticles(
        const moldyn::SimpleSphericalParticles& src) {
    *this = src;
}


/*
 * moldyn::SimpleSphericalParticles::~SimpleSphericalParticles
 */
moldyn::SimpleSphericalParticles::~SimpleSphericalParticles(void) {
    this->colDataType = COLDATA_NONE;
    this->colPtr = NULL; // DO NOT DELETE
    this->count = 0;
    this->vertDataType = VERTDATA_NONE;
    this->vertPtr = NULL; // DO NOT DELETE
}


/*
 * moldyn::SimpleSphericalParticles::operator=
 */
moldyn::SimpleSphericalParticles&
moldyn::SimpleSphericalParticles::operator=(
        const moldyn::SimpleSphericalParticles& rhs) {
    this->col[0] = rhs.col[0];
    this->col[1] = rhs.col[1];
    this->col[2] = rhs.col[2];
    this->col[3] = rhs.col[3];
    this->colDataType = rhs.colDataType;
    this->colPtr = rhs.colPtr;
    this->colStride = rhs.colStride;
    this->count = rhs.count;
    this->maxColI = rhs.maxColI;
    this->minColI = rhs.minColI;
    this->radius = rhs.radius;
    this->particleType = rhs.particleType;
    this->vertDataType = rhs.vertDataType;
    this->vertPtr = rhs.vertPtr;
    this->vertStride = rhs.vertStride;
	this->disabledNullChecks = rhs.disabledNullChecks;
	this->clusterInfos = rhs.clusterInfos;
    return *this;
}


/*
 * moldyn::SimpleSphericalParticles::operator==
 */
bool moldyn::SimpleSphericalParticles::operator==(
        const moldyn::SimpleSphericalParticles& rhs) const {
    return ((this->col[0] == rhs.col[0])
        && (this->col[1] == rhs.col[1])
        && (this->col[2] == rhs.col[2])
        && (this->col[3] == rhs.col[3])
        && (this->colDataType == rhs.colDataType)
        && (this->colPtr == rhs.colPtr)
        && (this->colStride == rhs.colStride)
        && (this->count == rhs.count)
        && (this->maxColI == rhs.maxColI)
        && (this->minColI == rhs.minColI)
        && (this->radius == rhs.radius)
        && (this->vertDataType == rhs.vertDataType)
        && (this->vertPtr == rhs.vertPtr)
        && (this->vertStride == rhs.vertStride)
		&& (this->clusterInfos == rhs.clusterInfos));
}

/****************************************************************************/


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
