/*
 * AbstractSimpleStoringParticleDataSource.cpp
 *
 * Copyright (C) 2012 by TU Dresden (CGV)
 * Alle Rechte vorbehalten.
 */

#include "AbstractSimpleStoringParticleDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"

namespace megamol::datatools::io {


/*
 * AbstractSimpleStoringParticleDataSource::AbstractSimpleStoringParticleDataSource
 */
AbstractSimpleStoringParticleDataSource::AbstractSimpleStoringParticleDataSource(void)
        : AbstractSimpleParticleDataSource()
        , posData()
        , posDataType(geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE)
        , colData()
        , colDataType(geocalls::MultiParticleDataCall::Particles::COLDATA_NONE)
        , bbox()
        , cbox()
        , defCol(192, 192, 192, 255)
        , defRad(0.5f)
        , minColVal(0.0f)
        , maxColVal(1.0f)
        , datahash(0) {
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
            if (!this->loadData(this->filenameSlot.Param<core::param::FilePathParam>()->Value().string().c_str())) {
                throw vislib::Exception("Failed to load data from file", __FILE__, __LINE__);
            }
            this->datahash++;
        } catch (vislib::Exception ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Exception: %s [%s, %d]\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        } catch (...) { megamol::core::utility::log::Log::DefaultLog.WriteError("Unexpected exception"); }
    }
}


/*
 * AbstractSimpleStoringParticleDataSource::getData
 */
bool AbstractSimpleStoringParticleDataSource::getData(geocalls::MultiParticleDataCall& call) {
    this->assertData();

    call.AccessBoundingBoxes().Clear();
    call.AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    call.AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);

    call.SetDataHash(this->datahash);

    call.SetFrameCount(1);
    call.SetFrameID(0);

    if (this->posData.IsEmpty())
        return false;

    call.SetParticleListCount(1);
    geocalls::SimpleSphericalParticles& p = call.AccessParticles(0);

    SIZE_T size;
    switch (this->posDataType) {
    case geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
        size = sizeof(float) * 3;
        break;
    case geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
        size = sizeof(float) * 4;
        break;
    case geocalls::SimpleSphericalParticles::VERTDATA_SHORT_XYZ:
        size = sizeof(unsigned short) * 3;
        break;
    default:
        return false;
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
bool AbstractSimpleStoringParticleDataSource::getExtent(geocalls::MultiParticleDataCall& call) {
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
} // namespace megamol::datatools::io
