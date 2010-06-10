/*
 * DataFileSequence.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DataFileSequence.h"
#include "moldyn/MultiParticleDataCall.h"
#include "moldyn/ParticleGrilDataCall.h"
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

    this->fileNameTemplateSlot << new param::StringParam();
    this->MakeSlotAvailable(&this->fileNameTemplateSlot);

    this->fileNumberMinSlot << new param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->fileNumberMinSlot);

    this->fileNumberMaxSlot << new param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->fileNumberMaxSlot);

    this->fileNumberStepSlot << new param::IntParam(1, 1);
    this->MakeSlotAvailable(&this->fileNumberStepSlot);

    this->fileNameSlotNameSlot << new param::StringParam();
    this->MakeSlotAvailable(&this->fileNameSlotNameSlot);

    // TODO: Better enumerate compatible calls
    this->outDataSlot.SetCallback("MultiParticleDataCall", "GetData", &DataFileSequence::getDataCallback);
    this->outDataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &DataFileSequence::getExtentCallback);
    this->inDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->outDataSlot.SetCallback("ParticleGridDataCall", "GetData", &DataFileSequence::getDataCallback);
    this->outDataSlot.SetCallback("ParticleGridDataCall", "GetExtent", &DataFileSequence::getExtentCallback);
    this->inDataSlot.SetCompatibleCall<ParticleGridDataCallDescription>();

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
    //if (!this->filenameSlot.Param<param::FilePathParam>()->Value().IsEmpty()) {
    //    this->filenameChanged(this->filenameSlot);
    //}
    return true;
}


/*
 * moldyn::DataFileSequence::release
 */
void moldyn::DataFileSequence::release(void) {
    //this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    //this->data.EnforceSize(0);
}


/*
 * moldyn::DataFileSequence::getDataCallback
 */
bool moldyn::DataFileSequence::getDataCallback(Call& caller) {
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
    //MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    //if (c2 == NULL) return false;

    //c2->SetExtent(1,
    //    this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(),
    //    this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    //c2->SetDataHash(this->datahash);

    //return true;
    return false;
}
