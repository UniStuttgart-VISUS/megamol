/*
 * AbstractSimpleParticleDataSource.cpp
 *
 * Copyright (C) 2012 by TU Dresden (CGV)
 * Alle Rechte vorbehalten.
 */

#include "AbstractSimpleParticleDataSource.h"
#include "mmcore/param/FilePathParam.h"

namespace megamol::datatools::io {


/*
 * AbstractSimpleParticleDataSource::AbstractSimpleParticleDataSource
 */
AbstractSimpleParticleDataSource::AbstractSimpleParticleDataSource()
        : Module()
        , filenameSlot("filename", "Full path to the file to load")
        , getDataSlot("getdata", "Publishes data for other modules") {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->getDataSlot.SetCallback(
        "MultiParticleDataCall", "GetData", &AbstractSimpleParticleDataSource::getDataCallback);
    this->getDataSlot.SetCallback(
        "MultiParticleDataCall", "GetExtent", &AbstractSimpleParticleDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);
}


/*
 * AbstractSimpleParticleDataSource::AbstractSimpleParticleDataSource
 */
AbstractSimpleParticleDataSource::~AbstractSimpleParticleDataSource() {
    this->Release();
}


/*
 * AbstractSimpleParticleDataSource::getDataCallback
 */
bool AbstractSimpleParticleDataSource::getDataCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (mpdc == NULL)
        return false;
    return this->getData(*mpdc);
}


/*
 * AbstractSimpleParticleDataSource::getDataCallback
 */
bool AbstractSimpleParticleDataSource::getExtentCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (mpdc == NULL)
        return false;
    return this->getExtent(*mpdc);
}
} // namespace megamol::datatools::io
