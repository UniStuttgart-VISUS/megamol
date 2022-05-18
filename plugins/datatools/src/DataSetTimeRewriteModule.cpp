/*
 * DataSetTimeRewriteModule.cpp
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */

#include "DataSetTimeRewriteModule.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/param/IntParam.h"
#include "stdafx.h"
#include <algorithm>

using namespace megamol;


/*
 * datatools::DataSetTimeRewriteModule::DataSetTimeRewriteModule
 */
datatools::DataSetTimeRewriteModule::DataSetTimeRewriteModule(void)
        : core::Module()
        , outDataSlot("outData", "The slot for publishing data to the writer")
        , inDataSlot("inData", "The slot for requesting data from the source")
        , firstFrameSlot("firstFrame", "The number of the first frame")
        , lastFrameSlot("lastFrame", "The number of the last frame")
        , frameStepSlot("frameStep", "The step length between two frames") {

    //core::CallDescriptionManager::DescriptionIterator iter(core::CallDescriptionManager::Instance()->GetIterator());
    //const core::CallDescription *cd = NULL;
    //while ((cd = this->moveToNextCompatibleCall(iter)) != NULL) {
    //    this->outDataSlot.SetCallback(cd->ClassName(), "GetData", &DataSetTimeRewriteModule::getDataCallback);
    //    this->outDataSlot.SetCallback(cd->ClassName(), "GetExtent", &DataSetTimeRewriteModule::getExtentCallback);
    //    this->inDataSlot.SetCompatibleCall(*cd);
    //}

    //this->MakeSlotAvailable(&this->outDataSlot);
    //this->MakeSlotAvailable(&this->inDataSlot);

    this->firstFrameSlot.SetParameter(new core::param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->firstFrameSlot);

    this->lastFrameSlot.SetParameter(new core::param::IntParam(1000000, 0));
    this->MakeSlotAvailable(&this->lastFrameSlot);

    this->frameStepSlot.SetParameter(new core::param::IntParam(1, 1));
    this->MakeSlotAvailable(&this->frameStepSlot);
}


/*
 * datatools::DataSetTimeRewriteModule::~DataSetTimeRewriteModule
 */
datatools::DataSetTimeRewriteModule::~DataSetTimeRewriteModule(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * datatools::DataSetTimeRewriteModule::create
 */
bool datatools::DataSetTimeRewriteModule::create(void) {
    for (auto cd : this->GetCoreInstance()->GetCallDescriptionManager()) {
        if (IsCallDescriptionCompatible(cd)) {
            this->outDataSlot.SetCallback(cd->ClassName(), "GetData", &DataSetTimeRewriteModule::getDataCallback);
            this->outDataSlot.SetCallback(cd->ClassName(), "GetExtent", &DataSetTimeRewriteModule::getExtentCallback);
            this->inDataSlot.SetCompatibleCall(cd);
        }
    }
    this->MakeSlotAvailable(&this->outDataSlot);
    this->MakeSlotAvailable(&this->inDataSlot);
    return true;
}


/*
 * datatools::DataSetTimeRewriteModule::release
 */
void datatools::DataSetTimeRewriteModule::release(void) {}


/*
 * datatools::DataSetTimeRewriteModule::IsCallDescriptionCompatible
 */
bool datatools::DataSetTimeRewriteModule::IsCallDescriptionCompatible(core::factories::CallDescription::ptr desc) {
    return (desc->FunctionCount() == 2) && vislib::StringA("GetData").Equals(desc->FunctionName(0), false) &&
           vislib::StringA("GetExtent").Equals(desc->FunctionName(1), false);
}


/*
 * datatools::DataSetTimeRewriteModule::getDataCallback
 */
bool datatools::DataSetTimeRewriteModule::getDataCallback(core::Call& caller) {
    using megamol::core::AbstractGetData3DCall;
    if (!this->checkConnections(&caller))
        return false;

    AbstractGetData3DCall* pgdc = dynamic_cast<AbstractGetData3DCall*>(&caller);
    if (pgdc == NULL)
        return false;

    AbstractGetData3DCall* ggdc = this->inDataSlot.CallAs<AbstractGetData3DCall>();
    if (ggdc == NULL)
        return false;

    int minF = this->firstFrameSlot.Param<core::param::IntParam>()->Value();
    int maxF = this->lastFrameSlot.Param<core::param::IntParam>()->Value();
    int lenF = this->frameStepSlot.Param<core::param::IntParam>()->Value();

    this->GetCoreInstance()->GetCallDescriptionManager().AssignmentCrowbar(ggdc, pgdc);
    // Change time code
    ggdc->SetFrameID(minF + pgdc->FrameID() * lenF, pgdc->IsFrameForced());

    if (!(*ggdc)(0))
        return false;

    this->GetCoreInstance()->GetCallDescriptionManager().AssignmentCrowbar(pgdc, ggdc);
    ggdc->SetUnlocker(nullptr, false);
    // Change time code
    unsigned int fid = ggdc->FrameID();
    unsigned int fc = ggdc->FrameCount();
    unsigned int nfc = (std::min<unsigned int>(maxF, fc) - minF) / lenF;
    unsigned int nfid = (std::min<unsigned int>(maxF, fid) - minF) / lenF;
    pgdc->SetFrameCount(nfc);
    pgdc->SetFrameID(nfid);

    return true;
}


/*
 * datatools::DataSetTimeRewriteModule::getExtentCallback
 */
bool datatools::DataSetTimeRewriteModule::getExtentCallback(core::Call& caller) {
    using megamol::core::AbstractGetData3DCall;
    if (!this->checkConnections(&caller))
        return false;

    AbstractGetData3DCall* pgdc = dynamic_cast<AbstractGetData3DCall*>(&caller);
    if (pgdc == NULL)
        return false;

    AbstractGetData3DCall* ggdc = this->inDataSlot.CallAs<AbstractGetData3DCall>();
    if (ggdc == NULL)
        return false;

    int minF = this->firstFrameSlot.Param<core::param::IntParam>()->Value();
    int maxF = this->lastFrameSlot.Param<core::param::IntParam>()->Value();
    int lenF = this->frameStepSlot.Param<core::param::IntParam>()->Value();

    this->GetCoreInstance()->GetCallDescriptionManager().AssignmentCrowbar(ggdc, pgdc);
    // Change time code
    ggdc->SetFrameID(minF + pgdc->FrameID() * lenF);

    if (!(*ggdc)(1))
        return false;

    this->GetCoreInstance()->GetCallDescriptionManager().AssignmentCrowbar(pgdc, ggdc);
    ggdc->SetUnlocker(nullptr, false);
    // Change time code
    unsigned int fid = ggdc->FrameID();
    unsigned int fc = ggdc->FrameCount();
    unsigned int nfc = (std::min<unsigned int>(maxF, fc) - minF) / lenF;
    unsigned int nfid = (std::min<unsigned int>(maxF, fid) - minF) / lenF;
    pgdc->SetFrameCount(nfc);
    pgdc->SetFrameID(nfid);

    return true;
}


/*
 * datatools::DataSetTimeRewriteModule::checkConnections
 */
bool datatools::DataSetTimeRewriteModule::checkConnections(core::Call* outCall) {
    if (this->inDataSlot.GetStatus() != core::AbstractSlot::STATUS_CONNECTED)
        return false;
    if (this->outDataSlot.GetStatus() != core::AbstractSlot::STATUS_CONNECTED)
        return false;
    core::Call* inCall = this->inDataSlot.CallAs<core::Call>();
    if ((inCall == NULL) || (outCall == NULL))
        return false;
    for (auto cd : this->GetCoreInstance()->GetCallDescriptionManager()) {
        if (IsCallDescriptionCompatible(cd)) {
            if (cd->IsDescribing(inCall) && cd->IsDescribing(outCall))
                return true;
        }
    }
    return false;
}
