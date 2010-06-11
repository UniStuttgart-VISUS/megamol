/*
 * DataFileSequence.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DataFileSequence.h"
#include "CallDescriptionManager.h"
#include "CallDescription.h"
#include "param/StringParam.h"
#include "param/IntParam.h"
#include "vislib/Log.h"
//#include "vislib/MemmappedFile.h"
//#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
//#include "vislib/sysfunctions.h"
//#include "vislib/VersionNumber.h"

using namespace megamol::core;


/*
 * moldyn::DataFileSequence::DataFileSequence
 */
moldyn::DataFileSequence::DataFileSequence(void) : Module(),
        fileNameTemplateSlot("fileNameTemplate", "The file name template"),
        fileNumberMinSlot("fileNumberMin", "Slot for the minimum file number"),
        fileNumberMaxSlot("fileNumberMax", "Slot for the maximum file number"),
        fileNumberStepSlot("fileNumberStep", "Slot for the file number increase step"),
        fileNameSlotNameSlot("fileNameSlotName", "The name of the data source file name parameter slot"),
        outDataSlot("outData", "The slot for publishing data to the writer"),
        inDataSlot("inData", "The slot for requesting data from the source"),
        clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), datahash(0) {

    this->fileNameTemplateSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->fileNameTemplateSlot);

    this->fileNumberMinSlot << new param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->fileNumberMinSlot);

    this->fileNumberMaxSlot << new param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->fileNumberMaxSlot);

    this->fileNumberStepSlot << new param::IntParam(1, 1);
    this->MakeSlotAvailable(&this->fileNumberStepSlot);

    this->fileNameSlotNameSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->fileNameSlotNameSlot);

    CallDescriptionManager::DescriptionIterator iter(CallDescriptionManager::Instance()->GetIterator());
    const CallDescription *cd = NULL;
    while ((cd = this->moveToNextCompatibleCall(iter)) != NULL) {
        this->outDataSlot.SetCallback(cd->ClassName(), "GetData", &DataFileSequence::getDataCallback);
        this->outDataSlot.SetCallback(cd->ClassName(), "GetExtent", &DataFileSequence::getExtentCallback);
        this->inDataSlot.SetCompatibleCall(*cd);
    }

    this->MakeSlotAvailable(&this->outDataSlot);
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * moldyn::DataFileSequence::~DataFileSequence
 */
moldyn::DataFileSequence::~DataFileSequence(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::DataFileSequence::create
 */
bool moldyn::DataFileSequence::create(void) {
    return true;
}


/*
 * moldyn::DataFileSequence::release
 */
void moldyn::DataFileSequence::release(void) {
}


/*
 * moldyn::DataFileSequence::moveToNextCompatibleCall
 */
const CallDescription* moldyn::DataFileSequence::moveToNextCompatibleCall(
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
 * moldyn::DataFileSequence::getDataCallback
 */
bool moldyn::DataFileSequence::getDataCallback(Call& caller) {
    if (!this->checkConnections(&caller)) return false;
    this->checkParameters();

    //MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    //if (c2 == NULL) return false;

    //c2->SetUnlocker(NULL);
    //c2->SetParticleListCount(1);
    //c2->AccessParticles(0).SetCount(this->data.GetSize() / 19);
    //if ((this->data.GetSize() / 19) > 0) {
    //    c2->AccessParticles(0).SetColourData(
    //        moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB,
    //        this->data.At(16), 19);
    //    c2->AccessParticles(0).SetVertexData(
    //        moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR,
    //        this->data, 19);
    //}
    //c2->SetDataHash(this->datahash);

    //return true;
    return false;
}


/*
 * moldyn::DataFileSequence::getExtentCallback
 */
bool moldyn::DataFileSequence::getExtentCallback(Call& caller) {
    if (!this->checkConnections(&caller)) return false;
    this->checkParameters();

    //MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    //if (c2 == NULL) return false;

    //c2->SetExtent(1,
    //    this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(),
    //    this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    //c2->SetDataHash(this->datahash);

    //return true;
    return false;
}


/*
 * moldyn::DataFileSequence::checkConnections
 */
bool moldyn::DataFileSequence::checkConnections(Call *outCall) {
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


/*
 * moldyn::DataFileSequence::checkParameters
 */
void moldyn::DataFileSequence::checkParameters(void) {
    if (this->fileNameTemplateSlot.IsDirty()
            || this->fileNumberMinSlot.IsDirty()
            || this->fileNumberMaxSlot.IsDirty()
            || this->fileNumberStepSlot.IsDirty()
            || this->fileNameSlotNameSlot.IsDirty()) {

        // TODO: Implement

    }
}
