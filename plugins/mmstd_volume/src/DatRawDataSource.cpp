/*
 * DatRawDataSource.cpp
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DatRawDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/String.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/VersionNumber.h"
#include "vislib/StringConverter.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * volume::DatRawDataSource::DatRawDataSource
 */
volume::DatRawDataSource::DatRawDataSource(void) : Module(),
        datFilenameSlot("datFilename", "The path to the Dat file to load."),
        getDataSlot("getdata", "Slot to request data from this data source."),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), data(0), datahash(0) {
            
    this->datFilenameSlot.SetParameter(new core::param::FilePathParam(""));
    this->datFilenameSlot.SetUpdateCallback(&DatRawDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->datFilenameSlot);
    
    this->getDataSlot.SetCallback(core::moldyn::VolumeDataCall::ClassName(), 
        core::moldyn::VolumeDataCall::FunctionName(core::moldyn::VolumeDataCall::CallForGetData),
        &DatRawDataSource::getDataCallback);
    this->getDataSlot.SetCallback(core::moldyn::VolumeDataCall::ClassName(), 
        core::moldyn::VolumeDataCall::FunctionName(core::moldyn::VolumeDataCall::CallForGetExtent),
        &DatRawDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * volume::DatRawDataSource::~DatRawDataSource
 */
volume::DatRawDataSource::~DatRawDataSource(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * volume::DatRawDataSource::create
 */
bool volume::DatRawDataSource::create(void) {
    if (!this->datFilenameSlot.Param<core::param::FilePathParam>()->Value().IsEmpty()) {
        this->filenameChanged(this->datFilenameSlot);
    }
    return true;
}


/*
 * volume::DatRawDataSource::release
 */
void volume::DatRawDataSource::release(void) {
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
}


/*
 * volume::DatRawDataSource::filenameChanged
 */
bool volume::DatRawDataSource::filenameChanged(core::param::ParamSlot& slot) {
    using vislib::sys::Log;
   
    // try to load the header (0 means loading failed)
#ifdef WIN32
    if (datRaw_readHeader(this->datFilenameSlot.Param<core::param::FilePathParam>()->Value(), &this->header, NULL) == 0) {
        return false;
    }
#else
    if (datRaw_readHeader( this->datFilenameSlot.Param<core::param::FilePathParam>()->Value(), &this->header, NULL) == 0) {
        return false;
    }
#endif

    // load data
    this->data.AssertSize(datRaw_getBufferSize(&this->header, DR_FORMAT_FLOAT));
    void *tmpData = (void*)this->data;
    datRaw_getNext(&this->header, &tmpData, DR_FORMAT_FLOAT);

    // TODO calc bounding box
    //if (this->data.GetSize() >= bpp) {
    //    float *ptr = this->data.As<float>();
    //    float rad = 0.0f;
    //    if (this->verNum == 100) rad = ptr[3];
    //    this->bbox.Set(
    //        ptr[0] - rad, ptr[1] - rad, ptr[2] - rad,
    //        ptr[0] + rad, ptr[1] + rad, ptr[2] + rad);

    //    for (unsigned int i = bpp; i < this->data.GetSize(); i += bpp) {
    //        ptr = this->data.AsAt<float>(i);
    //        if (this->verNum == 100) rad = ptr[3];
    //        this->bbox.GrowToPoint(ptr[0] - rad, ptr[1] - rad, ptr[2] - rad);
    //        this->bbox.GrowToPoint(ptr[0] + rad, ptr[1] + rad, ptr[2] + rad);
    //    }

    //} else {
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    //}

    this->datahash++; // so simple, it might work

    return true; // to reset the dirty flag of the param slot
}


/*
 * volume::DatRawDataSource::getDataCallback
 */
bool volume::DatRawDataSource::getDataCallback(core::Call& caller) {
    core::moldyn::VolumeDataCall *vdc = dynamic_cast<core::moldyn::VolumeDataCall*>(&caller);
    if (vdc == NULL) return false;

    vdc->SetUnlocker(NULL);
    
    /*
    float min, max;
    if (this->header.resolution[0] * this->header.resolution[1] * this->header.resolution[2] > 0) {
        min = max = this->data.As<float>()[0];
        for (int i = 1; i < this->header.resolution[0] * this->header.resolution[1] * this->header.resolution[2]; i++) {
            min = min < this->data.As<float>()[i] ? min : this->data.As<float>()[i];
            max = max > this->data.As<float>()[i] ? max : this->data.As<float>()[i];
        }
    }
    // DEBUG normalize data
    for (unsigned int i = 0; i < this->header.resolution[0] * this->header.resolution[1] * this->header.resolution[2]; i++) {
        this->data.As<float>()[i] = (this->data.As<float>()[i] - min) / (max - min);
    }
    */

    // set data
    vdc->SetBoundingBox(this->bbox);
    vdc->SetVolumeDimension(this->header.resolution[0], this->header.resolution[1], this->header.resolution[2]);
    vdc->SetVoxelMapPointer(this->data.As<float>());
    //vdc->SetMinimumDensity(min);
    //vdc->SetMaximumDensity(max);

    vdc->SetDataHash(this->datahash);

    return true;
}


/*
 * volume::DatRawDataSource::getExtentCallback
 */
bool volume::DatRawDataSource::getExtentCallback(core::Call& caller) {
    core::moldyn::VolumeDataCall *vdc = dynamic_cast<core::moldyn::VolumeDataCall*>(&caller);
    if (vdc == NULL) return false;

    // TODO fix frame count
    vdc->SetExtent(1,
        this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(),
        this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    vdc->SetDataHash(this->datahash);

    return true;
}
