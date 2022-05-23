/*
 * AbstractTriMeshDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "AbstractTriMeshDataSource.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/assert.h"

using namespace megamol;
using namespace megamol::trisoup_gl;


/*
 * AbstractTriMeshDataSource::AbstractTriMeshDataSource
 */
AbstractTriMeshDataSource::AbstractTriMeshDataSource(void)
        : core::Module()
        , objs()
        , mats()
        , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , datahash(0)
        , getDataSlot("getdata", "The slot publishing the loaded data")
        , getLinesDataSlot("getLinesData", "The slot publishing loaded lines data") {

    this->getDataSlot.SetCallback(
        megamol::geocalls_gl::CallTriMeshDataGL::ClassName(), "GetData", &AbstractTriMeshDataSource::getDataCallback);
    this->getDataSlot.SetCallback(megamol::geocalls_gl::CallTriMeshDataGL::ClassName(), "GetExtent",
        &AbstractTriMeshDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getLinesDataSlot.SetCallback(
        megamol::geocalls::LinesDataCall::ClassName(), "GetData", &AbstractTriMeshDataSource::getDataCallback);
    this->getLinesDataSlot.SetCallback(
        megamol::geocalls::LinesDataCall::ClassName(), "GetExtent", &AbstractTriMeshDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getLinesDataSlot);
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
    megamol::geocalls_gl::CallTriMeshDataGL* ctmd = dynamic_cast<megamol::geocalls_gl::CallTriMeshDataGL*>(&caller);
    megamol::geocalls::LinesDataCall* ldc = dynamic_cast<megamol::geocalls::LinesDataCall*>(&caller);
    if (ctmd == NULL && ldc == NULL)
        return false;
    this->assertData();

    if (ctmd != NULL) {
        ctmd->SetDataHash(this->datahash);
        ctmd->SetObjects(static_cast<unsigned int>(this->objs.Count()), this->objs.PeekElements());
        ctmd->SetUnlocker(NULL);
    } else if (ldc != NULL) {
        ldc->SetDataHash(this->datahash);
        ldc->SetData(static_cast<unsigned int>(this->lines.size()), this->lines.data());
        ldc->SetUnlocker(NULL);
    }

    return true;
}


/*
 * AbstractTriMeshDataSource::getExtentCallback
 */
bool AbstractTriMeshDataSource::getExtentCallback(core::Call& caller) {
    megamol::geocalls_gl::CallTriMeshDataGL* ctmd = dynamic_cast<megamol::geocalls_gl::CallTriMeshDataGL*>(&caller);
    megamol::geocalls::LinesDataCall* ldc = dynamic_cast<megamol::geocalls::LinesDataCall*>(&caller);
    if (ctmd == NULL && ldc == NULL)
        return false;
    this->assertData();

    if (ctmd != NULL) {
        ctmd->SetDataHash(this->datahash);
        ctmd->SetExtent(1, this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(), this->bbox.Right(),
            this->bbox.Top(), this->bbox.Front());
    } else if (ldc != NULL) {
        ldc->SetDataHash(this->datahash);
        ldc->SetExtent(1, this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(), this->bbox.Right(),
            this->bbox.Top(), this->bbox.Front());
    }

    return true;
}
