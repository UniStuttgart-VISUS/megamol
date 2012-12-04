/*
 * DatRawDataSource.cpp
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DatRawDataSource.h"
#include "param/FilePathParam.h"
#include "param/FloatParam.h"
#include "param/StringParam.h"
#include "vislib/Log.h"
#include "vislib/MemmappedFile.h"
#include "vislib/String.h"
#include "vislib/sysfunctions.h"
#include "vislib/VersionNumber.h"
#include "vislib/StringConverter.h"


using namespace megamol::core;


/*
 * moldyn::DatRawDataSource::DatRawDataSource
 */
moldyn::DatRawDataSource::DatRawDataSource(void) : Module(),
        datFilenameSlot("datFilename", "The path to the Dat file to load."),
        getDataSlot("getdata", "Slot to request data from this data source."),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), data(0), datahash(0) {
            
    this->datFilenameSlot.SetParameter(new param::FilePathParam(""));
    this->datFilenameSlot.SetUpdateCallback(&DatRawDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->datFilenameSlot);
    
    this->getDataSlot.SetCallback( VolumeDataCall::ClassName(), 
        VolumeDataCall::FunctionName(VolumeDataCall::CallForGetData),
        &DatRawDataSource::getDataCallback);
    this->getDataSlot.SetCallback( VolumeDataCall::ClassName(), 
        VolumeDataCall::FunctionName(VolumeDataCall::CallForGetExtent),
        &DatRawDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * moldyn::DatRawDataSource::~DatRawDataSource
 */
moldyn::DatRawDataSource::~DatRawDataSource(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::DatRawDataSource::create
 */
bool moldyn::DatRawDataSource::create(void) {
    if (!this->datFilenameSlot.Param<param::FilePathParam>()->Value().IsEmpty() ) {
        this->filenameChanged(this->datFilenameSlot);
    }
    return true;
}


/*
 * moldyn::DatRawDataSource::release
 */
void moldyn::DatRawDataSource::release(void) {
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
}


/*
 * moldyn::DatRawDataSource::filenameChanged
 */
bool moldyn::DatRawDataSource::filenameChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
   
    // try to load the header (0 means loading failed)
    if( datRaw_readHeader( W2A(this->datFilenameSlot.Param<param::FilePathParam>()->Value()), &this->header, NULL) == 0 ) {
        return false;
    }

    // load data
    this->data.AssertSize( datRaw_getBufferSize( &this->header, DR_FORMAT_FLOAT));
    void *tmpData = (void*)this->data;
    datRaw_getNext( &this->header, &tmpData, DR_FORMAT_FLOAT);

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
 * moldyn::DatRawDataSource::getDataCallback
 */
bool moldyn::DatRawDataSource::getDataCallback(Call& caller) {
    VolumeDataCall *vdc = dynamic_cast<VolumeDataCall*>(&caller);
    if ( vdc == NULL) return false;

    vdc->SetUnlocker(NULL);
    
    /*
    float min, max;
    if( this->header.resolution[0] * this->header.resolution[1] * this->header.resolution[2] > 0 ) {
        min = max = this->data.As<float>()[0];
        for( int i = 1; i < this->header.resolution[0] * this->header.resolution[1] * this->header.resolution[2]; i++ ) {
            min = min < this->data.As<float>()[i] ? min : this->data.As<float>()[i];
            max = max > this->data.As<float>()[i] ? max : this->data.As<float>()[i];
        }
    }
    // DEBUG normalize data
    for( unsigned int i = 0; i < this->header.resolution[0] * this->header.resolution[1] * this->header.resolution[2]; i++ ) {
        this->data.As<float>()[i] = ( this->data.As<float>()[i] - min) / ( max - min);
    }
    */

    // set data
    vdc->SetBoundingBox( this->bbox);
    vdc->SetVolumeDimension( this->header.resolution[0], this->header.resolution[1], this->header.resolution[2]);
    vdc->SetVoxelMapPointer( this->data.As<float>());
    //vdc->SetMinimumDensity( min);
    //vdc->SetMaximumDensity( max);

    vdc->SetDataHash(this->datahash);

    return true;
}


/*
 * moldyn::DatRawDataSource::getExtentCallback
 */
bool moldyn::DatRawDataSource::getExtentCallback(Call& caller) {
    VolumeDataCall *vdc = dynamic_cast<VolumeDataCall*>(&caller);
    if ( vdc == NULL) return false;

    // TODO fix frame count
    vdc->SetExtent(1,
        this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(),
        this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    vdc->SetDataHash(this->datahash);

    return true;
}
