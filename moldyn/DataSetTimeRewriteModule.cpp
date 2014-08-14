/*
 * DataSetTimeRewriteModule.cpp
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DataSetTimeRewriteModule.h"
#include "AbstractGetData3DCall.h"
#include "CallDescriptionManager.h"
#include "CallDescription.h"
#include "param/IntParam.h"

using namespace megamol::core;


/*
 * moldyn::DataSetTimeRewriteModule::DataSetTimeRewriteModule
 */
moldyn::DataSetTimeRewriteModule::DataSetTimeRewriteModule(void) : Module(),
        outDataSlot("outData", "The slot for publishing data to the writer"),
        inDataSlot("inData", "The slot for requesting data from the source"),
        firstFrameSlot("firstFrame", "The number of the first frame"),
        lastFrameSlot("lastFrame", "The number of the last frame"),
        frameStepSlot("frameStep", "The step length between two frames") {

    CallDescriptionManager::DescriptionIterator iter(CallDescriptionManager::Instance()->GetIterator());
    const CallDescription *cd = NULL;
    while ((cd = this->moveToNextCompatibleCall(iter)) != NULL) {
        this->outDataSlot.SetCallback(cd->ClassName(), "GetData", &DataSetTimeRewriteModule::getDataCallback);
        this->outDataSlot.SetCallback(cd->ClassName(), "GetExtent", &DataSetTimeRewriteModule::getExtentCallback);
        this->inDataSlot.SetCompatibleCall(*cd);
    }

    this->MakeSlotAvailable(&this->outDataSlot);
    this->MakeSlotAvailable(&this->inDataSlot);

    this->firstFrameSlot.SetParameter(new param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->firstFrameSlot);

    this->lastFrameSlot.SetParameter(new param::IntParam(1000000, 0));
    this->MakeSlotAvailable(&this->lastFrameSlot);

    this->frameStepSlot.SetParameter(new param::IntParam(1, 1));
    this->MakeSlotAvailable(&this->frameStepSlot);
}


/*
 * moldyn::DataSetTimeRewriteModule::~DataSetTimeRewriteModule
 */
moldyn::DataSetTimeRewriteModule::~DataSetTimeRewriteModule(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::DataSetTimeRewriteModule::create
 */
bool moldyn::DataSetTimeRewriteModule::create(void) {
    return true;
}


/*
 * moldyn::DataSetTimeRewriteModule::release
 */
void moldyn::DataSetTimeRewriteModule::release(void) {
}


/*
 * moldyn::DataSetTimeRewriteModule::moveToNextCompatibleCall
 */
const CallDescription* moldyn::DataSetTimeRewriteModule::moveToNextCompatibleCall(
        CallDescriptionManager::DescriptionIterator &iterator) const {
    while (iterator.HasNext()) {
        const CallDescription *d = iterator.Next();
        if ((d->FunctionCount() == 2)
                && vislib::StringA("GetData").Equals(d->FunctionName(0), false)
                && vislib::StringA("GetExtent").Equals(d->FunctionName(1), false)) {
            return d;
        }
    }
    return NULL;
}


/*
 * moldyn::DataSetTimeRewriteModule::getDataCallback
 */
bool moldyn::DataSetTimeRewriteModule::getDataCallback(Call& caller) {
    if (!this->checkConnections(&caller)) return false;

    AbstractGetData3DCall *pgdc = dynamic_cast<AbstractGetData3DCall*>(&caller);
    if (pgdc == NULL) return false;

    AbstractGetData3DCall *ggdc = this->inDataSlot.CallAs<AbstractGetData3DCall>();
    if (ggdc == NULL) return false;

    int minF = this->firstFrameSlot.Param<param::IntParam>()->Value();
    int maxF = this->lastFrameSlot.Param<param::IntParam>()->Value();
    int lenF = this->frameStepSlot.Param<param::IntParam>()->Value();

    CallDescriptionManager::Instance()->AssignmentCrowbar(ggdc, pgdc);
    // Change time code
    ggdc->SetFrameID(minF + pgdc->FrameID() * lenF);

    if (!(*ggdc)(0)) return false;

    CallDescriptionManager::Instance()->AssignmentCrowbar(pgdc, ggdc);
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
 * moldyn::DataSetTimeRewriteModule::getExtentCallback
 */
bool moldyn::DataSetTimeRewriteModule::getExtentCallback(Call& caller) {
    if (!this->checkConnections(&caller)) return false;

    AbstractGetData3DCall *pgdc = dynamic_cast<AbstractGetData3DCall*>(&caller);
    if (pgdc == NULL) return false;

    AbstractGetData3DCall *ggdc = this->inDataSlot.CallAs<AbstractGetData3DCall>();
    if (ggdc == NULL) return false;

    int minF = this->firstFrameSlot.Param<param::IntParam>()->Value();
    int maxF = this->lastFrameSlot.Param<param::IntParam>()->Value();
    int lenF = this->frameStepSlot.Param<param::IntParam>()->Value();

    CallDescriptionManager::Instance()->AssignmentCrowbar(ggdc, pgdc);
    // Change time code
    ggdc->SetFrameID(minF + pgdc->FrameID() * lenF);

    if (!(*ggdc)(1)) return false;

    CallDescriptionManager::Instance()->AssignmentCrowbar(pgdc, ggdc);
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
 * moldyn::DataSetTimeRewriteModule::checkConnections
 */
bool moldyn::DataSetTimeRewriteModule::checkConnections(Call *outCall) {
    if (this->inDataSlot.GetStatus() != AbstractSlot::STATUS_CONNECTED) return false;
    if (this->outDataSlot.GetStatus() != AbstractSlot::STATUS_CONNECTED) return false;
    Call *inCall = this->inDataSlot.CallAs<Call>();
    if ((inCall == NULL) || (outCall == NULL)) return false;
    CallDescriptionManager::DescriptionIterator iter(CallDescriptionManager::Instance()->GetIterator());
    const CallDescription *cd = NULL;
    while ((cd = this->moveToNextCompatibleCall(iter)) != NULL) {
        if (cd->IsDescribing(inCall) && cd->IsDescribing(outCall)) return true;
        // both slot connected with similar calls
    }
    return false;
}
