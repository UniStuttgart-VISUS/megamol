/*
 * mmvtkmFileLoader.cpp
 *
 * Copyright (C) 2020-2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */


#include "mmvtkm/mmvtkmFileLoader.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

#include "vtkm/io/reader/VTKDataSetReader.h"


using namespace megamol;
using namespace megamol::mmvtkm;


/*
 * mmvtkmFileLoader::mmvtkmFileLoader
 */
mmvtkmFileLoader::mmvtkmFileLoader(void)
        : core::Module()
        , getDataCalleeSlot_("getdata", "Slot to request data from this data source.")
        , filename_("filename", "The path to the vtkm file to load.")
        , version_(0)
        , vtkmData_()
        , vtkmMetaData_()
        , vtkmDataFile_("")
        , fileChanged_(false) {
    this->filename_.SetParameter(new core::param::FilePathParam(""));
    this->filename_.SetUpdateCallback(&mmvtkmFileLoader::filenameChanged);
    this->MakeSlotAvailable(&this->filename_);


    this->getDataCalleeSlot_.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(0),
        &mmvtkmFileLoader::getDataCallback); // GetData is FunctionName(0)
    this->getDataCalleeSlot_.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(1),
        &mmvtkmFileLoader::getMetaDataCallback); // GetExtent is FunctionName(1)
    this->MakeSlotAvailable(&this->getDataCalleeSlot_);
}


/*
 * mmvtkmFileLoader::~mmvtkmFileLoader
 */
mmvtkmFileLoader::~mmvtkmFileLoader(void) {
    this->Release();
}


/*
 * mmvtkmFileLoader::create
 */
bool mmvtkmFileLoader::create(void) {
    return true;
}


/*
 * mmvtkmFileLoader::release
 */
void mmvtkmFileLoader::release(void) {}


/*
 * mmvtkmFileLoader::filenameChanged
 */
bool mmvtkmFileLoader::filenameChanged(core::param::ParamSlot& slot) {
    vtkmDataFile_ = this->filename_.Param<core::param::FilePathParam>()->ValueString();

    if (vtkmDataFile_.empty()) {
        core::utility::log::Log::DefaultLog.WriteError("Empty vtkm file!");
        return false;
    }


    try {
        vtkm::io::reader::VTKDataSetReader dataReader(vtkmDataFile_);
        vtkmData_ = std::make_shared<VtkmData>();
        vtkmData_->data = dataReader.ReadDataSet();
        vtkmMetaData_.minMaxBounds = vtkmData_->data.GetCoordinateSystem(0).GetBounds();

        core::utility::log::Log::DefaultLog.WriteInfo("File successfully loaded.");
    } catch (const std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError("In % s at line %d. \n", __FILE__, __LINE__);
        core::utility::log::Log::DefaultLog.WriteError(e.what());
        return false;
    }


    fileChanged_ = true;


    return true;
}


/*
 * mmvtkmFileLoader::getDataCallback
 */
bool mmvtkmFileLoader::getDataCallback(core::Call& caller) {
    mmvtkmDataCall* lhsVtkmDc = dynamic_cast<mmvtkmDataCall*>(&caller);
    if (lhsVtkmDc == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError("In %s at line %d. lhsVtkmDc is nullptr.", __FILE__, __LINE__);
        return false;
    }


    if (fileChanged_) {
        lhsVtkmDc->setData(vtkmData_, ++version_);
        lhsVtkmDc->setMetaData(vtkmMetaData_);

        //filename_.ResetDirty();
        fileChanged_ = false;

        return true;
    }


    return true;
}


/*
 * mmvtkmFileLoader::getMetaDataCallback
 */
bool mmvtkmFileLoader::getMetaDataCallback(core::Call& caller) {
    mmvtkmDataCall* lhsVtkmDc = dynamic_cast<mmvtkmDataCall*>(&caller);

    if (fileChanged_) {
        vtkmMetaData_.minMaxBounds = vtkmData_->data.GetCoordinateSystem(0).GetBounds();
        lhsVtkmDc->setMetaData(vtkmMetaData_);
    }


    return true;
}
