//
// TrajectorySmoothFilter.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 8, 2013
//     Author: scharnkn
//


#include <omp.h>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "protein_calls/MolecularDataCall.h"

#include "TrajectorySmoothFilter.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;


/*
 * TrajectorySmoothFilter::TrajectorySmoothFilter
 */
TrajectorySmoothFilter::TrajectorySmoothFilter()
        : core::Module()
        , molDataCallerSlot("getdata", "Connects the filter with molecule data storage")
        , dataOutSlot("dataout", "The slot providing the filtered data")
        , nAvgFramesSlot("nAvgFrames", "Number of frames to average over") {

    // Enable caller slot
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataCallerSlot);

    // Enable dataout slot
    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &TrajectorySmoothFilter::getData);
    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &TrajectorySmoothFilter::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // Set number of averaging frames
    this->nAvgFrames = 10;
    this->nAvgFramesSlot.SetParameter(new param::IntParam(this->nAvgFrames, 1));
    this->MakeSlotAvailable(&this->nAvgFramesSlot);
}


/*
 * TrajectorySmoothFilter::~TrajectorySmoothFilter
 */
TrajectorySmoothFilter::~TrajectorySmoothFilter() {
    this->Release();
}


/*
 * TrajectorySmoothFilter::create
 */
bool TrajectorySmoothFilter::create() {
    return true;
}


/*
 * TrajectorySmoothFilter::release
 */
void TrajectorySmoothFilter::release() {}


/*
 * TrajectorySmoothFilter::getData
 */
bool TrajectorySmoothFilter::getData(megamol::core::Call& call) {
    using megamol::core::utility::log::Log;

    uint firstFrame = 0;

    // Get a pointer to the outgoing data call
    MolecularDataCall* molOut = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (molOut == NULL) {
        return false;
    }

    // Get a pointer to the incoming data call
    MolecularDataCall* molIn = dynamic_cast<MolecularDataCall*>(&call);
    if (molIn == NULL) {
        return false;
    }

    firstFrame = molIn->FrameID();

    // Obtain number of atoms
    molOut->SetFrameID(0, true); // Set 'force' flag
    if (!(*molOut)(MolecularDataCall::CallForGetData)) {
        return false;
    }
    molOut->Unlock();

    if (molIn->FrameID() >= (molOut->FrameCount() - (this->nAvgFrames - 1))) {
        return false;
    }

    // (Re-)allocate memory and init with zero
    this->atomPosSmoothed.Validate(molOut->AtomCount() * 3);
    memset(this->atomPosSmoothed.Peek(), 0, molOut->AtomCount() * 3 * sizeof(float));

    // Loop through all averaging frames
    for (uint fr = 0; fr < this->nAvgFrames; ++fr) {
        molOut->SetFrameID(molIn->FrameID() + fr, true); // Set 'force' flag
        if (!(*molOut)(MolecularDataCall::CallForGetData)) {
            return false;
        }
        molOut->Unlock();

        // Add positions
#pragma omp parallel for
        for (int at = 0; at < static_cast<int>(molOut->AtomCount()); ++at) {
            this->atomPosSmoothed.Peek()[3 * at + 0] += molOut->AtomPositions()[3 * at + 0];
            this->atomPosSmoothed.Peek()[3 * at + 1] += molOut->AtomPositions()[3 * at + 1];
            this->atomPosSmoothed.Peek()[3 * at + 2] += molOut->AtomPositions()[3 * at + 2];
        }
    }

    // Normalize positions
#pragma omp parallel for
    for (int at = 0; at < static_cast<int>(molOut->AtomCount()); ++at) {
        this->atomPosSmoothed.Peek()[3 * at + 0] /= static_cast<float>(this->nAvgFrames);
        this->atomPosSmoothed.Peek()[3 * at + 1] /= static_cast<float>(this->nAvgFrames);
        this->atomPosSmoothed.Peek()[3 * at + 2] /= static_cast<float>(this->nAvgFrames);
    }

    // Transfer data from outgoing to incoming data call
    *molIn = *molOut;

    // Set new smoothed positions
    molIn->SetAtomPositions(this->atomPosSmoothed.Peek());

    // Correct extent, we have lesser frames because of the averging
    molIn->SetExtent(molOut->FrameCount() - (this->nAvgFrames - 1), molOut->AccessBoundingBoxes());

    molIn->SetFrameID(firstFrame); // Restore correct frame id

    // Set unlocker object for incoming data call
    molIn->SetUnlocker(new TrajectorySmoothFilter::Unlocker(*molOut));

    return true;
}


/*
 * TrajectorySmoothFilter::getExtent
 */
bool TrajectorySmoothFilter::getExtent(core::Call& call) {

    // Get a pointer to the incoming data call
    MolecularDataCall* molIn = dynamic_cast<MolecularDataCall*>(&call);
    if (molIn == NULL) {
        return false;
    }

    // Get a pointer to the outgoing data call
    MolecularDataCall* molOut = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (molOut == NULL) {
        return false;
    }

    // Get extend
    if (!(*molOut)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }

    // Set extent, the filter module outputs less frames than the original
    // data module because of the averaging
    molIn->AccessBoundingBoxes().Clear();
    molIn->SetExtent(molOut->FrameCount() - (this->nAvgFrames - 1), molOut->AccessBoundingBoxes());

    return true;
}


/*
 * TrajectorySmoothFilter::updateParams
 */
void TrajectorySmoothFilter::updateParams(MolecularDataCall* mol) {
    // Parameter to determine number of averaging frames
    if (this->nAvgFramesSlot.IsDirty()) {
        this->nAvgFrames = this->nAvgFramesSlot.Param<core::param::BoolParam>()->Value();
        this->nAvgFramesSlot.ResetDirty();
    }
}
