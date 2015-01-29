/*
 * ParticleGridDataCall.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "moldyn/ParticleGridDataCall.h"
//#include "vislib/memutils.h"

using namespace megamol::core;

/****************************************************************************/


/*
 * moldyn::ParticleGridDataCall::ParticleType::ParticleType
 */
moldyn::ParticleGridDataCall::ParticleType::ParticleType(void)
        : colDataType(moldyn::MultiParticleDataCall::Particles::COLDATA_NONE),
        maxColI(1.0f), minColI(0.0), radius(0.5f),
        vertDataType(moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
    this->col[0] = this->col[1] = this->col[2] = 128;
}


/*
 * moldyn::ParticleGridDataCall::ParticleType::ParticleType
 */
moldyn::ParticleGridDataCall::ParticleType::ParticleType(
        const moldyn::ParticleGridDataCall::ParticleType& src)
        : colDataType(src.colDataType), maxColI(src.maxColI),
        minColI(src.minColI), radius(src.radius),
        vertDataType(src.vertDataType) {
    this->col[0] = src.col[0];
    this->col[1] = src.col[1];
    this->col[2] = src.col[2];
}


/*
 * moldyn::ParticleGridDataCall::ParticleType::~ParticleType
 */
moldyn::ParticleGridDataCall::ParticleType::~ParticleType(void) {
    // intentionally empty
}


/*
 * moldyn::ParticleGridDataCall::ParticleType::operator=
 */
moldyn::ParticleGridDataCall::ParticleType&
moldyn::ParticleGridDataCall::ParticleType::operator=(
        const moldyn::ParticleGridDataCall::ParticleType& rhs) {
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
 * moldyn::ParticleGridDataCall::ParticleType::operator==
 */
bool moldyn::ParticleGridDataCall::ParticleType::operator==(
        const moldyn::ParticleGridDataCall::ParticleType& rhs) const {
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
 * moldyn::ParticleGridDataCall::Particles::Particles
 */
moldyn::ParticleGridDataCall::Particles::Particles(void) : colPtr(NULL),
        colStride(0), count(0), maxRad(0.5f), vertPtr(NULL), vertStride(0) {
    // intentionally empty
}


/*
 * moldyn::ParticleGridDataCall::Particles::Particles
 */
moldyn::ParticleGridDataCall::Particles::Particles(
        const moldyn::ParticleGridDataCall::Particles& src)
        : colPtr(src.colPtr), colStride(src.colStride), count(src.count),
        maxRad(src.maxRad), vertPtr(src.vertPtr), vertStride(src.vertStride) {
    // intentionally empty
}


/*
 * moldyn::ParticleGridDataCall::Particles::~Particles
 */
moldyn::ParticleGridDataCall::Particles::~Particles(void) {
    this->colPtr = NULL; // DO NOT DELETE
    this->vertPtr = NULL; // DO NOT DELETE
}


/*
 * moldyn::ParticleGridDataCall::Particles::operator=
 */
moldyn::ParticleGridDataCall::Particles&
moldyn::ParticleGridDataCall::Particles::operator=(
        const moldyn::ParticleGridDataCall::Particles& rhs) {
    this->colPtr = rhs.colPtr;
    this->colStride = rhs.colStride;
    this->count = rhs.count;
    this->maxRad = rhs.maxRad;
    this->vertPtr = rhs.vertPtr;
    this->vertStride = rhs.vertStride;
    return *this;
}


/*
 * moldyn::ParticleGridDataCall::Particles::operator==
 */
bool moldyn::ParticleGridDataCall::Particles::operator==(
        const moldyn::ParticleGridDataCall::Particles& rhs) const {
    return (this->colPtr == rhs.colPtr)
        && (this->colStride == rhs.colStride)
        && (this->count == rhs.count)
        && (this->maxRad == rhs.maxRad)
        && (this->vertPtr == rhs.vertPtr)
        && (this->vertStride == rhs.vertStride);
}

/****************************************************************************/

/*
 * moldyn::ParticleGridDataCall::GridCell::GridCell
 */
moldyn::ParticleGridDataCall::GridCell::GridCell(void) : particles(NULL),
        bbox() {
    // intentionally empty
}


/*
 * moldyn::ParticleGridDataCall::GridCell::GridCell
 */
moldyn::ParticleGridDataCall::GridCell::GridCell(
        const moldyn::ParticleGridDataCall::GridCell& src) : particles(NULL),
        bbox() {
    *this = src;
}


/*
 * moldyn::ParticleGridDataCall::GridCell::~GridCell
 */
moldyn::ParticleGridDataCall::GridCell::~GridCell(void) {
    ARY_SAFE_DELETE(this->particles);
}


/*
 * moldyn::ParticleGridDataCall::GridCell::operator=
 */
moldyn::ParticleGridDataCall::GridCell&
moldyn::ParticleGridDataCall::GridCell::operator=(
        const moldyn::ParticleGridDataCall::GridCell& rhs) {
    this->bbox = rhs.bbox;
    return *this;
}


/*
 * moldyn::ParticleGridDataCall::GridCell::operator==
 */
bool moldyn::ParticleGridDataCall::GridCell::operator==(
        const moldyn::ParticleGridDataCall::GridCell& rhs) const {
    return this->bbox == rhs.bbox;
}


/****************************************************************************/

/*
 * moldyn::ParticleGridDataCall::ParticleGridDataCall
 */
moldyn::ParticleGridDataCall::ParticleGridDataCall(void)
        : AbstractGetData3DCall(), cntCellsX(0), cntCellsY(0), cntCellsZ(0),
        cntCells(0), cells(NULL), ownCellMem(false), cntTypes(0),
        types(NULL), ownTypeMem(false) {
    // Intentionally empty
}


/*
 * moldyn::ParticleGridDataCall::~ParticleGridDataCall
 */
moldyn::ParticleGridDataCall::~ParticleGridDataCall(void) {
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
 * moldyn::ParticleGridDataCall::operator=
 */
moldyn::ParticleGridDataCall& moldyn::ParticleGridDataCall::operator=(
        const moldyn::ParticleGridDataCall& rhs) {
    AbstractGetData3DCall::operator=(rhs);

    this->SetGridDataRef(rhs.cntCellsX, rhs.cntCellsY, rhs.cntCellsZ, rhs.cells);
    this->SetTypeDataRef(rhs.cntTypes, rhs.types);

    return *this;
}
