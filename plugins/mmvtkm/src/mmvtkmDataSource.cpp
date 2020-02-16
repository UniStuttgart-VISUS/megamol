/*
 * mmvtkmDataSource.cpp (MMPLDDataSource)
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmvtkm/mmvtkmDataSource.h"
#include "CallADIOSData.cpp"
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
    , getData("getdata", "Slot to request data from this data source.")
    , nodesAdiosCallerSlot("adiosSlot", "Slot to request data from adios.")
    , labelAdiosCallerSlot("adiosSlot", "Slot to request data from adios.")
    , filename("filename", "The path to the vtkm file to load.")
    , topology("topology", "The path to the tetrahedron file to load.")
    , labels("labels", "The path to the node labels file to load.")
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

    this->topology.SetParameter(new core::param::FilePathParam(""));
    this->topology.SetUpdateCallback(&mmvtkmDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->topology);

    this->labels.SetParameter(new core::param::FilePathParam(""));
    this->labels.SetUpdateCallback(&mmvtkmDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->labels);

    this->getData.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(0),
        &mmvtkmDataSource::getDataCallback); // GetData is FunctionName(0)
    this->getData.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(1),
        &mmvtkmDataSource::getMetaDataCallback); // GetExtent is FunctionName(1)
    this->MakeSlotAvailable(&this->getData);

    this->nodesAdiosCallerSlot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->nodesAdiosCallerSlot);

    this->labelAdiosCallerSlot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->labelAdiosCallerSlot);
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
        // f->Clear();
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

    adios::CallADIOSData* nodesCad = this->nodesAdiosCallerSlot.CallAs<adios::CallADIOSData>();
    if (nodesCad == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("nodesCallADIOSData is nullptr");
        return false;
    }

    adios::CallADIOSData* labelCad = this->labelAdiosCallerSlot.CallAs<adios::CallADIOSData>();
    if (labelCad == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("labelCallADIOSData is nullptr");
        return false;
    }

    /////////////////////////////
    // get tetrahedron coordinates from csv file through adios
    /////////////////////////////

    // TODO: look what happens detailed
    // TODO: place it somewhere else? or check if data has changed from callee
    // getHeaderCallback
    if (!(*nodesCad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error during nodes getExtents");
        return false;
    }

    if (!(*labelCad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error during label getExtents");
        return false;
    }

    // getDataCallback
    if (!(*nodesCad)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error during nodes getData");
        return false;
    }

    if (!(*labelCad)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error during label getData");
        return false;
    }


    std::string coord_type = nodesCad->getData("x")->getType();
    std::string label_type = nodesCad->getData("node-label")->getType();

    // this assumes the data type from the file is float
    // TODO do for general case
    std::vector<float> node_labels = nodesCad->getData("node-label")->GetAsFloat();
    std::vector<float> x_coord = nodesCad->getData("x")->GetAsFloat();
    std::vector<float> y_coord = nodesCad->getData("y")->GetAsFloat();
    std::vector<float> z_coord = nodesCad->getData("z")->GetAsFloat();

    std::vector<float> element_label = labelCad->getData("element-label")->GetAsFloat();
    std::vector<float> nodeA = labelCad->getData("NodeA")->GetAsFloat();
    std::vector<float> nodeB = labelCad->getData("NodeB")->GetAsFloat();
    std::vector<float> nodeC = labelCad->getData("NodeC")->GetAsFloat();
    std::vector<float> nodeD = labelCad->getData("NodeD")->GetAsFloat();

    for (int i = 0; i < element_label.size(); ++i) {
        std::vector<int> labels = {(int)nodeA[i], (int)nodeB[i], (int)nodeC[i], (int)nodeD[i]};
        std::vector < vtkm::Vec32F_32 > vertices(4);
        for (int j = 0; j < 4; ++j) {
            auto it = std::find(node_labels.begin(), node_labels.end(), labels[j]);
            int node_index = std::distance(node_labels.begin(), it);
            vertices[j] = {x_coord[node_index], y_coord[node_index], z_coord[node_index]};
        }

        // BUILD TETRAHEDRON HERE
    }


    // update data only when we have a new file
    if (this->filename.IsDirty()) {
        c2->SetDataHash(this->data_hash);
        c2->SetDataSet(&vtkmData);
        filename.ResetDirty();

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
