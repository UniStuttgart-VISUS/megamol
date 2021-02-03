/*
 * mmvtkmDataSource.cpp (MMPLDDataSource)
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmvtkm/mmvtkmDataSource.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmvtkm/mmvtkmDataCall.h"
#include "vislib/String.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"
#include "vtkm/io/reader/VTKDataSetReader.h"
#include "vtkm/io/reader/VTKPolyDataReader.h"


using namespace megamol;
using namespace megamol::mmvtkm;

/*
 * moldyn::mmvtkmDataSource::mmvtkmDataSource
 */
mmvtkmDataSource::mmvtkmDataSource(void)
    : core::view::AnimDataModule()
    , filename("filename", "The path to the vtkm file to load.")
    , getData("getdata", "Slot to request data from this data source.")
    , file(NULL)
    , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
    , clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
    , data_hash(0)
    , vtkmDataFile("")
    , vtkmData()
    , dirtyFlag(true) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&mmvtkmDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(0),
        &mmvtkmDataSource::getDataCallback); // GetData is FunctionName(0) moldyn::mmvtkmDataCall::FunctionName(0)
    this->getData.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(1),
        &mmvtkmDataSource::getMetaDataCallback); // GetExtent is FunctionName(1)
                                                 // moldyn::mmvtkmDataCall::FunctionName(1);
    this->MakeSlotAvailable(&this->getData);
}


/*
 * moldyn::mmvtkmDataSource::~mmvtkmDataSource
 */
mmvtkmDataSource::~mmvtkmDataSource(void) { this->Release(); }
/*
 moldyn::mmvtkmDataSource::constructFrame 
 */
core::view::AnimDataModule::Frame* mmvtkmDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<mmvtkmDataSource*>(this));
    return f;
}


/*
 * moldyn::mmvtkmDataSource::create
 */
bool mmvtkmDataSource::create(void) { return true; }

/*
 * moldyn::mmvtkmDataSource::loadFrame
 */
void mmvtkmDataSource::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {
    using vislib::sys::Log;
    Frame* f = dynamic_cast<Frame*>(frame);
    if (f == NULL) return;
    if (this->file == NULL) {
        //f->Clear();
        return;
    }
    // printf("Requesting frame %u of %u frames\n", idx, this->FrameCount());
    // Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Requesting frame %u of %u frames\n", idx, this->FrameCount());
    ASSERT(idx < this->FrameCount());

    // if (!f->LoadFrame(this->file, idx, this->frameIdx[idx + 1] - this->frameIdx[idx], this->fileVersion)) {
    //    // failed
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from vtkm file\n", idx);
    //}
}


/*
 * moldyn::mmvtkmDataSource::release
 */
void mmvtkmDataSource::release(void) {
    this->resetFrameCache();
    if (this->file != NULL) {
        vislib::sys::File* f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
}

/*
 * moldyn::mmvtkmDataSource::filenameChanged
 */
bool mmvtkmDataSource::filenameChanged(core::param::ParamSlot& slot) {
    this->data_hash++;

    vtkmDataFile = this->filename.Param<core::param::FilePathParam>()->ValueString();
    if (vtkmDataFile.empty()) {
        vislib::sys::Log::DefaultLog.WriteInfo("Empty vtkm file!");
    }

   vislib::sys::Log::DefaultLog.WriteInfo(
        "If no \"Safety check\" is shown, something went wrong reading the data. Probably the necessary line "
        "is not commented out. See readme");

   vtkm::io::reader::VTKDataSetReader readData(vtkmDataFile);
   vtkmData = readData.ReadDataSet();
   vislib::sys::Log::DefaultLog.WriteInfo("Safety check");

    dirtyFlag = true;

    return true;
}


/*
 * moldyn::mmvtkmDataSource::getDataCallback
 */
bool mmvtkmDataSource::getDataCallback(core::Call& caller) {
    mmvtkmDataCall* c2 = dynamic_cast<mmvtkmDataCall*>(&caller);
    if (c2 == NULL) return false;

    // update data only when we have a new file
    if (/*this->filename.IsDirty()*/ dirtyFlag) {
        c2->SetDataHash(this->data_hash);
        c2->SetDataSet(&vtkmData);
        dirtyFlag = false;
        // filename.ResetDirty();

        return true;
    }

    return false;
}


/*
 * moldyn::mmvtkmDataSource::getMetaDataCallback
 */
bool mmvtkmDataSource::getMetaDataCallback(core::Call& caller) {
    mmvtkmDataCall* c2 = dynamic_cast<mmvtkmDataCall*>(&caller);

    if (c2 != NULL && dirtyFlag) {
        c2->SetDataHash(this->data_hash);
        return true;
    }

    return false;
}
