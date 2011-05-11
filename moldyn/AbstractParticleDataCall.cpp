/*
 * AbstractParticleDataCall.cpp
 *
 * Copyright (C) VISUS 2011 (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractParticleDataCall.h"
//#include "vislib/memutils.h"


/*
 * Intentionally empty
 */

//using namespace megamol::core;
//
//
//
///****************************************************************************/
//
//
///*
// * moldyn::AbstractParticleDataCall::Particles::Particles
// */
//moldyn::AbstractParticleDataCall::Particles::Particles(void)
//        : colDataType(COLDATA_NONE), colPtr(NULL), colStride(0), count(0),
//        maxColI(1.0f), minColI(0.0f), radius(0.5f),
//        vertDataType(VERTDATA_NONE), vertPtr(NULL), vertStride(0) {
//    this->col[0] = 255;
//    this->col[1] = 0;
//    this->col[2] = 0;
//    this->col[3] = 255;
//}
//
//
///*
// * moldyn::AbstractParticleDataCall::Particles::Particles
// */
//moldyn::AbstractParticleDataCall::Particles::Particles(
//        const moldyn::AbstractParticleDataCall::Particles& src) {
//    *this = src;
//}
//
//
///*
// * moldyn::AbstractParticleDataCall::Particles::~Particles
// */
//moldyn::AbstractParticleDataCall::Particles::~Particles(void) {
//    this->colDataType = COLDATA_NONE;
//    this->colPtr = NULL; // DO NOT DELETE
//    this->count = 0;
//    this->vertDataType = VERTDATA_NONE;
//    this->vertPtr = NULL; // DO NOT DELETE
//}
//
//
///*
// * moldyn::AbstractParticleDataCall::Particles::operator=
// */
//moldyn::AbstractParticleDataCall::Particles&
//moldyn::AbstractParticleDataCall::Particles::operator=(
//        const moldyn::AbstractParticleDataCall::Particles& rhs) {
//    this->col[0] = rhs.col[0];
//    this->col[1] = rhs.col[1];
//    this->col[2] = rhs.col[2];
//    this->col[3] = rhs.col[3];
//    this->colDataType = rhs.colDataType;
//    this->colPtr = rhs.colPtr;
//    this->colStride = rhs.colStride;
//    this->count = rhs.count;
//    this->maxColI = rhs.maxColI;
//    this->minColI = rhs.minColI;
//    this->radius = rhs.radius;
//    this->vertDataType = rhs.vertDataType;
//    this->vertPtr = rhs.vertPtr;
//    this->vertStride = rhs.vertStride;
//    return *this;
//}
//
//
///*
// * moldyn::AbstractParticleDataCall::Particles::operator==
// */
//bool moldyn::AbstractParticleDataCall::Particles::operator==(
//        const moldyn::AbstractParticleDataCall::Particles& rhs) const {
//    return ((this->col[0] == rhs.col[0])
//        && (this->col[1] == rhs.col[1])
//        && (this->col[2] == rhs.col[2])
//        && (this->col[3] == rhs.col[3])
//        && (this->colDataType == rhs.colDataType)
//        && (this->colPtr == rhs.colPtr)
//        && (this->colStride == rhs.colStride)
//        && (this->count == rhs.count)
//        && (this->maxColI == rhs.maxColI)
//        && (this->minColI == rhs.minColI)
//        && (this->radius == rhs.radius)
//        && (this->vertDataType == rhs.vertDataType)
//        && (this->vertPtr == rhs.vertPtr)
//        && (this->vertStride == rhs.vertStride));
//}
//
///****************************************************************************/
