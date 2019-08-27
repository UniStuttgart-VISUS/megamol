/*
 * ParticleGridDataCall.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParticleGridDataCall.h"
//#include "vislib/memutils.h"

using namespace megamol::stdplugin::moldyn::rendering;

/****************************************************************************/


/*
 * ParticleGridDataCall::ParticleType::ParticleType
 */
ParticleGridDataCall::ParticleType::ParticleType(void)
        : colDataType(core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE),
        maxColI(1.0f), minColI(0.0), radius(0.5f),
        vertDataType(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
    this->col[0] = this->col[1] = this->col[2] = 128;
}


/*
 * ParticleGridDataCall::ParticleType::ParticleType
 */
ParticleGridDataCall::ParticleType::ParticleType(
        const ParticleGridDataCall::ParticleType& src)
        : colDataType(src.colDataType), maxColI(src.maxColI),
        minColI(src.minColI), radius(src.radius),
        vertDataType(src.vertDataType) {
    this->col[0] = src.col[0];
    this->col[1] = src.col[1];
    this->col[2] = src.col[2];
}


/*
 * ParticleGridDataCall::ParticleType::~ParticleType
 */
ParticleGridDataCall::ParticleType::~ParticleType(void) {
    // intentionally empty
}


/*
 * ParticleGridDataCall::ParticleType::operator=
 */
ParticleGridDataCall::ParticleType&
ParticleGridDataCall::ParticleType::operator=(
        const ParticleGridDataCall::ParticleType& rhs) {
    this->col[0] = rhs.col[0];
    this->col[1] = rhs.col[1];
    this->col[2] = rhs.col[2];
    this->colDataType = rhs.colDataType;
    this->maxColI = rhs.maxColI;
    this->minColI = rhs.minColI;
    this->radius = rhs.radius;
    this->vertDataType = rhs.vertDataType;
    return *this;
}


/*
 * ParticleGridDataCall::ParticleType::operator==
 */
bool ParticleGridDataCall::ParticleType::operator==(
        const ParticleGridDataCall::ParticleType& rhs) const {
    return (this->col[0] == rhs.col[0])
        && (this->col[1] == rhs.col[1])
        && (this->col[2] == rhs.col[2])
        && (this->colDataType == rhs.colDataType)
        && (this->maxColI == rhs.maxColI)
        && (this->minColI == rhs.minColI)
        && (this->radius == rhs.radius)
        && (this->vertDataType == rhs.vertDataType);
}

/****************************************************************************/


/*
 * ParticleGridDataCall::Particles::Particles
 */
ParticleGridDataCall::Particles::Particles(void) : colPtr(NULL),
        colStride(0), count(0), maxRad(0.5f), vertPtr(NULL), vertStride(0) {
    // intentionally empty
}


/*
 * ParticleGridDataCall::Particles::Particles
 */
ParticleGridDataCall::Particles::Particles(
        const ParticleGridDataCall::Particles& src)
        : colPtr(src.colPtr), colStride(src.colStride), count(src.count),
        maxRad(src.maxRad), vertPtr(src.vertPtr), vertStride(src.vertStride) {
    // intentionally empty
}


/*
 * ParticleGridDataCall::Particles::~Particles
 */
ParticleGridDataCall::Particles::~Particles(void) {
    this->colPtr = NULL; // DO NOT DELETE
    this->vertPtr = NULL; // DO NOT DELETE
}


/*
 * ParticleGridDataCall::Particles::operator=
 */
ParticleGridDataCall::Particles&
ParticleGridDataCall::Particles::operator=(
        const ParticleGridDataCall::Particles& rhs) {
    this->colPtr = rhs.colPtr;
    this->colStride = rhs.colStride;
    this->count = rhs.count;
    this->maxRad = rhs.maxRad;
    this->vertPtr = rhs.vertPtr;
    this->vertStride = rhs.vertStride;
    return *this;
}


/*
 * ParticleGridDataCall::Particles::operator==
 */
bool ParticleGridDataCall::Particles::operator==(
        const ParticleGridDataCall::Particles& rhs) const {
    return (this->colPtr == rhs.colPtr)
        && (this->colStride == rhs.colStride)
        && (this->count == rhs.count)
        && (this->maxRad == rhs.maxRad)
        && (this->vertPtr == rhs.vertPtr)
        && (this->vertStride == rhs.vertStride);
}

/****************************************************************************/

/*
 * ParticleGridDataCall::GridCell::GridCell
 */
ParticleGridDataCall::GridCell::GridCell(void) : particles(NULL),
        bbox() {
    // intentionally empty
}


/*
 * ParticleGridDataCall::GridCell::GridCell
 */
ParticleGridDataCall::GridCell::GridCell(
        const ParticleGridDataCall::GridCell& src) : particles(NULL),
        bbox() {
    *this = src;
}


/*
 * ParticleGridDataCall::GridCell::~GridCell
 */
ParticleGridDataCall::GridCell::~GridCell(void) {
    ARY_SAFE_DELETE(this->particles);
}


/*
 * ParticleGridDataCall::GridCell::operator=
 */
ParticleGridDataCall::GridCell&
ParticleGridDataCall::GridCell::operator=(
        const ParticleGridDataCall::GridCell& rhs) {
    this->bbox = rhs.bbox;
    return *this;
}


/*
 * ParticleGridDataCall::GridCell::operator==
 */
bool ParticleGridDataCall::GridCell::operator==(
        const ParticleGridDataCall::GridCell& rhs) const {
    return this->bbox == rhs.bbox;
}


/****************************************************************************/

/*
 * ParticleGridDataCall::ParticleGridDataCall
 */
ParticleGridDataCall::ParticleGridDataCall(void)
        : AbstractGetData3DCall(), cntCellsX(0), cntCellsY(0), cntCellsZ(0),
        cntCells(0), cells(NULL), ownCellMem(false), cntTypes(0),
        types(NULL), ownTypeMem(false) {
    // Intentionally empty
}


/*
 * ParticleGridDataCall::~ParticleGridDataCall
 */
ParticleGridDataCall::~ParticleGridDataCall(void) {
    this->Unlock();
    if (this->ownCellMem) {
        delete[] this->cells;
    }
    this->cells = NULL;
    this->cntCellsX = this->cntCellsY = this->cntCellsZ = this->cntCells = 0;
    if (this->ownTypeMem) {
        delete[] this->types;
    }
    this->types = NULL;
    this->cntTypes = 0;
}


/*
 * ParticleGridDataCall::operator=
 */
ParticleGridDataCall& ParticleGridDataCall::operator=(
        const ParticleGridDataCall& rhs) {
    AbstractGetData3DCall::operator=(rhs);

    this->SetGridDataRef(rhs.cntCellsX, rhs.cntCellsY, rhs.cntCellsZ, rhs.cells);
    this->SetTypeDataRef(rhs.cntTypes, rhs.types);

    return *this;
}
