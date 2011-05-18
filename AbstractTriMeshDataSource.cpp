/*
 * AbstractTriMeshDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "AbstractTriMeshDataSource.h"
#include "vislib/assert.h"
#include "vislib/Log.h"

using namespace megamol;
using namespace megamol::trisoup;


/*
 * AbstractTriMeshDataSource::AbstractTriMeshDataSource
 */
AbstractTriMeshDataSource::AbstractTriMeshDataSource(void) : core::Module(),
        objs(), mats(), bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), datahash(0),
        getDataSlot("getdata", "The slot publishing the loaded data") {

    this->getDataSlot.SetCallback(CallTriMeshData::ClassName(), "GetData", &AbstractTriMeshDataSource::getDataCallback);
    this->getDataSlot.SetCallback(CallTriMeshData::ClassName(), "GetExtent", &AbstractTriMeshDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * AbstractTriMeshDataSource::~AbstractTriMeshDataSource
 */
AbstractTriMeshDataSource::~AbstractTriMeshDataSource(void) {
    this->Release();
    ASSERT(this->objs.IsEmpty());
    ASSERT(this->mats.IsEmpty());
}


/*
 * AbstractTriMeshDataSource::create
 */
bool AbstractTriMeshDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * AbstractTriMeshDataSource::release
 */
void AbstractTriMeshDataSource::release(void) {
    this->objs.Clear();
    this->mats.Clear();
}


/*
 * AbstractTriMeshDataSource::getDataCallback
 */
bool AbstractTriMeshDataSource::getDataCallback(core::Call& caller) {
    CallTriMeshData *ctmd = dynamic_cast<CallTriMeshData *>(&caller);
    if (ctmd == NULL) return false;
    this->assertData();

    ctmd->SetDataHash(this->datahash);
    ctmd->SetObjects(static_cast<unsigned int>(this->objs.Count()), this->objs.PeekElements());
    ctmd->SetUnlocker(NULL);

    return true;
}


/*
 * AbstractTriMeshDataSource::getExtentCallback
 */
bool AbstractTriMeshDataSource::getExtentCallback(core::Call& caller) {
    CallTriMeshData *ctmd = dynamic_cast<CallTriMeshData *>(&caller);
    if (ctmd == NULL) return false;
    this->assertData();

    ctmd->SetDataHash(this->datahash);
    ctmd->SetExtent(1, this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(), this->bbox.Right(), this->bbox.Top(), this->bbox.Front());

    return true;
}
