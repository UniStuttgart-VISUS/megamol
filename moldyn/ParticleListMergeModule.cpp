/*
 * ParticleListMergeModule.cpp
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParticleListMergeModule.h"

using namespace megamol::core;


/*
 * moldyn::ParticleListMergeModule::ParticleListMergeModule
 */
moldyn::ParticleListMergeModule::ParticleListMergeModule(void) : Module(),
        outDataSlot("outData", "The slot for publishing data to the writer"),
        inDataSlot("inData", "The slot for requesting data from the source"),
        dataHash(0), frameId(0), parts(), data() {

    this->outDataSlot.SetCallback(MultiParticleDataCall::ClassName(), "GetData", &ParticleListMergeModule::getDataCallback);
    this->outDataSlot.SetCallback(MultiParticleDataCall::ClassName(), "GetExtent", &ParticleListMergeModule::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * moldyn::ParticleListMergeModule::~ParticleListMergeModule
 */
moldyn::ParticleListMergeModule::~ParticleListMergeModule(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::ParticleListMergeModule::create
 */
bool moldyn::ParticleListMergeModule::create(void) {
    return true;
}


/*
 * moldyn::ParticleListMergeModule::release
 */
void moldyn::ParticleListMergeModule::release(void) {
    // intentionally empty
}


/*
 * moldyn::ParticleListMergeModule::getDataCallback
 */
bool moldyn::ParticleListMergeModule::getDataCallback(Call& caller) {
    //if (!this->checkConnections(&caller)) return false;

    //AbstractGetData3DCall *pgdc = dynamic_cast<AbstractGetData3DCall*>(&caller);
    //if (pgdc == NULL) return false;

    //AbstractGetData3DCall *ggdc = this->inDataSlot.CallAs<AbstractGetData3DCall>();
    //if (ggdc == NULL) return false;

    //int minF = this->firstFrameSlot.Param<param::IntParam>()->Value();
    //int maxF = this->lastFrameSlot.Param<param::IntParam>()->Value();
    //int lenF = this->frameStepSlot.Param<param::IntParam>()->Value();

    //CallDescriptionManager::Instance()->AssignmentCrowbar(ggdc, pgdc);
    //// Change time code
    //ggdc->SetFrameID(minF + pgdc->FrameID() * lenF);

    //if (!(*ggdc)(0)) return false;

    //CallDescriptionManager::Instance()->AssignmentCrowbar(pgdc, ggdc);
    //ggdc->SetUnlocker(nullptr, false);
    //// Change time code
    //unsigned int fid = ggdc->FrameID();
    //unsigned int fc = ggdc->FrameCount();
    //unsigned int nfc = (std::min<unsigned int>(maxF, fc) - minF) / lenF;
    //unsigned int nfid = (std::min<unsigned int>(maxF, fid) - minF) / lenF;
    //pgdc->SetFrameCount(nfc);
    //pgdc->SetFrameID(nfid);

    //return true;

    return false;
}


/*
 * moldyn::ParticleListMergeModule::getExtentCallback
 */
bool moldyn::ParticleListMergeModule::getExtentCallback(Call& caller) {
    MultiParticleDataCall *pgdc = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (pgdc == NULL) return false;

    MultiParticleDataCall *ggdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (ggdc == NULL) return false;

    *ggdc = *pgdc;
    if (!(*ggdc)(1)) return false;

    *pgdc = *ggdc;
    ggdc->SetUnlocker(nullptr, false);

    return true;
}
