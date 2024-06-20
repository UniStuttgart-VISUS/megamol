/*
 * VolumetricDataSource.cpp
 *
 * Copyright (C) 2024 by Visualisierungsinstitut der Universit�t Stuttgart.
 * Alle rechte vorbehalten.
 */

#include "NewDatRawReader.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "datraw.h"
#include "mmcore/utility/log/Log.h"

megamol::volume::NewDatRawReader::NewDatRawReader()
        : core::Module()
        , slotGetData("GetData", "Slot for requesting data.")
        , paramFileName("FileName", "Path to file.") {
    using geocalls::VolumetricDataCall;
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), &NewDatRawReader::onGetData);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), &NewDatRawReader::onGetExtents);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), &NewDatRawReader::onGetMetadata);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_START_ASYNC), &NewDatRawReader::onStartAsync);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_STOP_ASYNC), &NewDatRawReader::onStopAsync);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_TRY_GET_DATA), &NewDatRawReader::onTryGetData);

    this->MakeSlotAvailable(&this->slotGetData);

    this->paramFileName.SetParameter(new core::param::FilePathParam(""));
    this->paramFileName.SetUpdateCallback(&NewDatRawReader::onFileNameChange);
    this->MakeSlotAvailable(&this->paramFileName);
}

megamol::volume::NewDatRawReader::~NewDatRawReader() {}

bool megamol::volume::NewDatRawReader::create() {
    return true;
}

void megamol::volume::NewDatRawReader::release() {}

bool megamol::volume::NewDatRawReader::onGetData(core::Call& call) {
    geocalls::VolumetricDataCall& c = dynamic_cast<geocalls::VolumetricDataCall&>(call);
    return true;
}

bool megamol::volume::NewDatRawReader::onGetMetadata(core::Call& call) {
    return false;
}

bool megamol::volume::NewDatRawReader::onGetExtents(core::Call& call) {
    try {
        std::vector<float> volExt;
        volExt.push_back(datInfo.slice_thickness()[0] * datInfo.resolution()[0]);
        volExt.push_back(datInfo.slice_thickness()[1] * datInfo.resolution()[1]);
        volExt.push_back(datInfo.slice_thickness()[2] * datInfo.resolution()[2]);

        geocalls::VolumetricDataCall& c = dynamic_cast<geocalls::VolumetricDataCall&>(call);
        c.SetExtent(datInfo.time_steps(), datInfo.origin()[0], datInfo.origin()[1], datInfo.origin()[2]
        , datInfo.origin()[0] + volExt[0], datInfo.origin()[1] + volExt[1], datInfo.origin()[2] + volExt[2]);
        return true;
    } catch (vislib::Exception e) {
        //TODO exception
        return false;
    }
}

bool megamol::volume::NewDatRawReader::onStartAsync(core::Call& call) {
    return true;
}

bool megamol::volume::NewDatRawReader::onStopAsync(core::Call& call) {
    return true;
}

bool megamol::volume::NewDatRawReader::onTryGetData(core::Call& call) {
    try {

    }
    catch (std::exception e) {
        geocalls::VolumetricDataCall& c = dynamic_cast<geocalls::VolumetricDataCall&>(call);
            //what does this do?
        c.SetDataHash(12345u);
        return false;
    }
    return false;
}

bool megamol::volume::NewDatRawReader::onFileNameChange(core::param::ParamSlot& slot) {
    // does this actually throw file not found exceptions?
    try {
        datInfo = datraw::info<char>::load(slot.Param<core::param::FilePathParam>()->Value().generic_string().c_str());
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("%s\n", e.what());
    }
    return true;
}
