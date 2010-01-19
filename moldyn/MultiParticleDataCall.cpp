/*
 * MultiParticleDataCall.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MultiParticleDataCall.h"
//#include "vislib/memutils.h"

using namespace megamol::core;

/****************************************************************************/


/*
 * moldyn::MultiParticleDataCall::Particles::Particles
 */
moldyn::MultiParticleDataCall::Particles::Particles(void)
        : colDataType(COLDATA_NONE), colPtr(NULL), colStride(0), count(0),
        maxColI(1.0f), minColI(0.0f), radius(0.5f),
        vertDataType(VERTDATA_NONE), vertPtr(NULL), vertStride(0) {
    this->col[0] = 255;
    this->col[1] = 0;
    this->col[2] = 0;
    this->col[3] = 255;
}


/*
 * moldyn::MultiParticleDataCall::Particles::Particles
 */
moldyn::MultiParticleDataCall::Particles::Particles(
        const moldyn::MultiParticleDataCall::Particles& src) {
    *this = src;
}


/*
 * moldyn::MultiParticleDataCall::Particles::~Particles
 */
moldyn::MultiParticleDataCall::Particles::~Particles(void) {
    this->colDataType = COLDATA_NONE;
    this->colPtr = NULL; // DO NOT DELETE
    this->count = 0;
    this->vertDataType = VERTDATA_NONE;
    this->vertPtr = NULL; // DO NOT DELETE
}


/*
 * moldyn::MultiParticleDataCall::Particles::operator=
 */
moldyn::MultiParticleDataCall::Particles&
moldyn::MultiParticleDataCall::Particles::operator=(
        const moldyn::MultiParticleDataCall::Particles& rhs) {
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
    this->vertDataType = rhs.vertDataType;
    this->vertPtr = rhs.vertPtr;
    this->vertStride = rhs.vertStride;
    return *this;
}


/*
 * moldyn::MultiParticleDataCall::Particles::operator==
 */
bool moldyn::MultiParticleDataCall::Particles::operator==(
        const moldyn::MultiParticleDataCall::Particles& rhs) const {
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
        && (this->vertStride == rhs.vertStride));
}

/****************************************************************************/


/*
 * moldyn::MultiParticleDataCall::MultiParticleDataCall
 */
moldyn::MultiParticleDataCall::MultiParticleDataCall(void)
        : AbstractGetData3DCall(), lists() {
    // Intentionally empty
}


/*
 * moldyn::MultiParticleDataCall::~MultiParticleDataCall
 */
moldyn::MultiParticleDataCall::~MultiParticleDataCall(void) {
    this->Unlock();
    this->lists.Clear();
}


/*
 * moldyn::MultiParticleDataCall::operator=
 */
moldyn::MultiParticleDataCall& moldyn::MultiParticleDataCall::operator=(
        const moldyn::MultiParticleDataCall& rhs) {
    AbstractGetData3DCall::operator =(rhs);
    this->lists.SetCount(rhs.lists.Count());
    for (SIZE_T i = 0; i < this->lists.Count(); i++) {
        this->lists[i] = rhs.lists[i];
    }
    return *this;
}
