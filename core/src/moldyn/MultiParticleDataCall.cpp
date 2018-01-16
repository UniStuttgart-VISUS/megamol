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

unsigned int megamol::core::moldyn::SimpleSphericalParticles::VertexDataSize[] = {0, 12, 16, 6};

unsigned int megamol::core::moldyn::SimpleSphericalParticles::ColorDataSize[] = {0, 3, 4, 12, 16, 4};

unsigned int megamol::core::moldyn::SimpleSphericalParticles::IDDataSize[] = {0, 4, 8};


/*
 * moldyn::SimpleSphericalParticles::SimpleSphericalParticles
 */
moldyn::SimpleSphericalParticles::SimpleSphericalParticles(void)
    : colDataType(COLDATA_NONE), colPtr(NULL), colStride(0), count(0)
    , maxColI(1.0f), minColI(0.0f), radius(0.5f), particleType(0)
    , vertDataType(VERTDATA_NONE), vertPtr(NULL), vertStride(0)
    , disabledNullChecks(false), clusterInfos(NULL)
    , idDataType{IDDATA_NONE}, idPtr{nullptr}, idStride{0} {
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
    this->idDataType = IDDATA_NONE;
    this->idPtr = nullptr;
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
    this->idDataType = rhs.idDataType;
    this->idPtr = rhs.idPtr;
    this->idStride = rhs.idStride;
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
        && (this->clusterInfos == rhs.clusterInfos)
        && (this->idDataType == rhs.idDataType)
        && (this->idPtr == rhs.idPtr)
        && (this->idStride == rhs.idStride));
}


/*
 * SimpleSphericalParticles::operator[]
 */
megamol::core::moldyn::SimpleSphericalParticles::particle_t const&
megamol::core::moldyn::SimpleSphericalParticles::operator[](size_t idx) const noexcept {
    /*return particle_t{
        this->vertPtr != nullptr ? static_cast<char const*>(this->vertPtr) + idx * this->vertStride : nullptr,
        this->colPtr != nullptr ? static_cast<char const*>(this->colPtr) + idx * this->colStride : nullptr,
        this->idPtr != nullptr ? static_cast<char const*>(this->idPtr) + idx * this->idStride : nullptr
    };*/
    return particle_t{
        VertexData_Base{this->vertexAccessor->Clone(),
            this->vertPtr != nullptr ? static_cast<char const*>(this->vertPtr) + idx * this->vertStride : nullptr},
        ColorData_Base{this->colorAccessor->Clone(),
            this->colPtr != nullptr ? static_cast<char const*>(this->colPtr) + idx * this->colStride : nullptr},
        IDData_Base{this->idAccessor->Clone(),
            this->idPtr != nullptr ? static_cast<char const*>(this->idPtr) + idx * this->idStride : nullptr}
    };
}


/*
 * SimpleSphericalParticles::At
 */
megamol::core::moldyn::SimpleSphericalParticles::particle_t const&
megamol::core::moldyn::SimpleSphericalParticles::At(size_t idx) const {
    if (idx < this->count) {
        return this->operator[](idx);
    } else {
        throw std::out_of_range("Idx larger than particle count.");
    }
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
