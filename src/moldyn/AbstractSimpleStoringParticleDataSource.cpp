/*
 * AbstractSimpleStoringParticleDataSource.cpp
 *
 * Copyright (C) 2012 by TU Dresden (CGV)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/moldyn/AbstractSimpleStoringParticleDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/Log.h"

using namespace megamol::core;
using namespace megamol::core::moldyn;


/*
 * AbstractSimpleStoringParticleDataSource::AbstractSimpleStoringParticleDataSource
 */
AbstractSimpleStoringParticleDataSource::AbstractSimpleStoringParticleDataSource(void) : AbstractSimpleParticleDataSource(),
        posData(), posDataType(MultiParticleDataCall::Particles::VERTDATA_NONE),
        colData(), colDataType(MultiParticleDataCall::Particles::COLDATA_NONE),
        bbox(), cbox(), defCol(192, 192, 192, 255), defRad(0.5f), minColVal(0.0f), maxColVal(1.0f), datahash(0) {
    this->filenameSlot.ForceSetDirty();
}


/*
 * AbstractSimpleStoringParticleDataSource::~AbstractSimpleStoringParticleDataSource
 */
AbstractSimpleStoringParticleDataSource::~AbstractSimpleStoringParticleDataSource(void) {
    this->Release();
}


/*
 * AbstractSimpleStoringParticleDataSource::assertData
 */
void AbstractSimpleStoringParticleDataSource::assertData(bool needLoad) {
    if (this->filenameSlot.IsDirty()) {
        needLoad = true;
        this->filenameSlot.ResetDirty();
    }
    if (needLoad) {
        try {
            if (!this->loadData(this->filenameSlot.Param<param::FilePathParam>()->Value())) {
                throw vislib::Exception("Failed to load data from file", __FILE__, __LINE__);
            }
            this->datahash++;
        } catch(vislib::Exception ex) {
            vislib::sys::Log::DefaultLog.WriteError("Exception: %s [%s, %d] (-> %s)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine(), ex.GetStack());
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteError("Unexpected exception");
        }
    }
}


/*
 * AbstractSimpleStoringParticleDataSource::getData
 */
bool AbstractSimpleStoringParticleDataSource::getData(MultiParticleDataCall& call) {
    this->assertData();

    call.AccessBoundingBoxes().Clear();
    call.AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    call.AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);

    call.SetDataHash(this->datahash);

    call.SetFrameCount(1);
    call.SetFrameID(0);

    if (this->posData.IsEmpty()) return false;

    call.SetParticleListCount(1);
    SimpleSphericalParticles &p = call.AccessParticles(0);

    SIZE_T size;
    switch (this->posDataType) {
    case SimpleSphericalParticles::VERTDATA_FLOAT_XYZ: size = sizeof(float) * 3; break;
    case SimpleSphericalParticles::VERTDATA_FLOAT_XYZR: size = sizeof(float) * 4; break;
    case SimpleSphericalParticles::VERTDATA_SHORT_XYZ: size = sizeof(unsigned short) * 3; break;
    default: return false;
    }

    p.SetCount(this->posData.GetSize() / size);
    p.SetColourData(this->colDataType, this->colData);
    p.SetColourMapIndexValues(this->minColVal, this->maxColVal);
    p.SetGlobalColour(this->defCol.R(), this->defCol.G(), this->defCol.B(), this->defCol.A());
    p.SetGlobalRadius(this->defRad);
    p.SetVertexData(this->posDataType, this->posData);

    call.SetUnlocker(NULL);

    return true;
}


/*
 * AbstractSimpleStoringParticleDataSource::getExtent
 */
bool AbstractSimpleStoringParticleDataSource::getExtent(MultiParticleDataCall& call) {
    this->assertData();

    call.AccessBoundingBoxes().Clear();
    call.AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    call.AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);

    call.SetDataHash(this->datahash);

    call.SetFrameCount(1);
    call.SetFrameID(0);

    call.SetUnlocker(NULL);

    return !this->posData.IsEmpty();
}
