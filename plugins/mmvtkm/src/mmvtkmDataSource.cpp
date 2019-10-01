/*
 * mmvtkmDataSource.cpp (MMPLDDataSource)
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vtkm/io/reader/VTKDataSetReader.h"
#include "mmvtkm/mmvtkmDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmvtkm/mmvtkmDataCall.h"
#include "mmcore/CoreInstance.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/FastFile.h"
#include "vislib/String.h"
#include "vislib/sys/SystemInformation.h"


using namespace megamol;
using namespace megamol::mmvtkm;

/*
 * moldyn::mmvtkmDataSource::Frame::Frame
 */
mmvtkmDataSource::Frame::Frame(core::view::AnimDataModule& owner)
        : core::view::AnimDataModule::Frame(owner), dat() {
    // intentionally empty
}


/*
 * moldyn::mmvtkmDataSource::Frame::~Frame
 */
mmvtkmDataSource::Frame::~Frame() {
    this->dat.EnforceSize(0);
}


/*
 * moldyn::mmvtkmDataSource::Frame::LoadFrame
 */
bool mmvtkmDataSource::Frame::LoadFrame(vislib::sys::File *file, unsigned int idx, UINT64 size, unsigned int version) {
    this->frame = idx;
    this->fileVersion = version;
    this->dat.EnforceSize(static_cast<SIZE_T>(size));
    return (file->Read(this->dat, size) == size);
}


/*
 * moldyn::mmvtkmDataSource::Frame::SetData
 */
void mmvtkmDataSource::Frame::SetData(mmvtkmDataCall& call) { 
}
/*****************************************************************************/


/*
 * moldyn::mmvtkmDataSource::mmvtkmDataSource
 */
mmvtkmDataSource::mmvtkmDataSource(void)
    : core::view::AnimDataModule()
    , filename("filename", "The path to the vtkm file to load.")
    , getData("getdata", "Slot to request data from this data source.")
    , file(NULL)
    , frameIdx(NULL)
    , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
    , clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
    , data_hash(0)
    , vtkmDataFile("")
    , currentFrame(0)
    , vtkmData()
    , dirtyFlag(true) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&mmvtkmDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(0),
        &mmvtkmDataSource::getDataCallback); // GetData is FunctionName(0) moldyn::mmvtkmDataCall::FunctionName(0)
    this->getData.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(1),
        &mmvtkmDataSource::getMetaDataCallback); // GetExtent is FunctionName(1) moldyn::mmvtkmDataCall::FunctionName(1);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * moldyn::mmvtkmDataSource::~mmvtkmDataSource
 */
mmvtkmDataSource::~mmvtkmDataSource(void) {
    this->Release();
}


/*
 * moldyn::mmvtkmDataSource::constructFrame
 */
core::view::AnimDataModule::Frame* mmvtkmDataSource::constructFrame(void) const {
    Frame *f = new Frame(*const_cast<mmvtkmDataSource*>(this));
    return f;
}


/*
 * moldyn::mmvtkmDataSource::create
 */
bool mmvtkmDataSource::create(void) {
	return true;
}


/*
 * moldyn::mmvtkmDataSource::loadFrame
 */
void mmvtkmDataSource::loadFrame(core::view::AnimDataModule::Frame* frame,
        unsigned int idx) {
    using vislib::sys::Log;
    Frame *f = dynamic_cast<Frame*>(frame);
    if (f == NULL) return;
    if (this->file == NULL) {
        f->Clear();
        return;
    }
    //printf("Requesting frame %u of %u frames\n", idx, this->FrameCount());
    //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Requesting frame %u of %u frames\n", idx, this->FrameCount());
    ASSERT(idx < this->FrameCount());

    //if (!f->LoadFrame(this->file, idx, this->frameIdx[idx + 1] - this->frameIdx[idx], this->fileVersion)) {
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
        vislib::sys::File *f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
    ARY_SAFE_DELETE(this->frameIdx);
}

/*
 * moldyn::mmvtkmDataSource::filenameChanged
 */
bool mmvtkmDataSource::filenameChanged(core::param::ParamSlot& slot) {
    this->data_hash++;

    vtkmDataFile = this->filename.Param<core::param::FilePathParam>()->ValueString();
	if (vtkmDataFile.empty()) {
        std::cout << "Empty vtkm file!" << '\n';
	}

    std::cout << "If no \"Safety check\" is shown, something went wrong reading the data. Probably the necessary line is not commented out. See readme" << '\n';
    vtkm::io::reader::VTKDataSetReader readData(vtkmDataFile);
    vtkmData = readData.ReadDataSet();
    std::cout << "Safety check" << '\n';

	dirtyFlag = true;

    return true;
}


/*
 * moldyn::mmvtkmDataSource::getDataCallback
 */
bool mmvtkmDataSource::getDataCallback(core::Call& caller) {
    mmvtkmDataCall *c2 = dynamic_cast<mmvtkmDataCall*>(&caller);
    if (c2 == NULL) return false;

	// update data only when we have a new file
	if (/*this->filename.IsDirty()*/ dirtyFlag) {
        // maybe reset frameid if file has changed
		c2->SetDataHash(this->data_hash);
		c2->SetDataSet(&vtkmData);
        //c2->SetFrameID(currentFrame++);
        dirtyFlag = false;
        //filename.ResetDirty();

		return true;
	}
    
	// if the file hasn't changed, no need to push data to renderer
	// only set next frame

    /*Frame *f = NULL;
    if (c2 != NULL) {
        f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID(), c2->IsFrameForced()));
        if (f == NULL) return false;
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetFrameID(f->FrameNumber());
        
        f->SetData(*c2);
    }*/

    return false;
}


/*
 * moldyn::mmvtkmDataSource::getMetaDataCallback
 */
bool mmvtkmDataSource::getMetaDataCallback(core::Call& caller) {
    mmvtkmDataCall *c2 = dynamic_cast<mmvtkmDataCall*>(&caller);
    if (c2 != NULL && dirtyFlag) {
		c2->SetFrameCount(this->FrameCount());
        c2->SetDataHash(this->data_hash);
        return true;
    }

    return false;
}
