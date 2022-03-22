/*
 * MolecularNeighborhood.cpp
 *
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * Author: Karsten Schatz
 * all rights reserved.
 */

#include "MolecularNeighborhood.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/math/Point.h"

#include "protein/GridNeighbourFinder.h"

#include <chrono>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 * MolecularNeighborhood::MolecularNeighborhood
 */
MolecularNeighborhood::MolecularNeighborhood(void)
        : Module()
        , getDataSlot("getData", "Calls molecular data")
        , dataOutSlot("dataOut", "Provides the molecular data with additional neighborhood information")
        , neighRadiusParam("radius", "The search radius for the neighborhood") {

    // caller slot
    this->getDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    // callee slot
    this->dataOutSlot.SetCallback(
        MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(0), &MolecularNeighborhood::getData);
    this->dataOutSlot.SetCallback(
        MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(1), &MolecularNeighborhood::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // other parameters
    this->neighRadiusParam.SetParameter(new param::FloatParam(10.0f, 0.0f, 20.0f));
    this->MakeSlotAvailable(&this->neighRadiusParam);

    this->lastDataHash = 0;
}

/*
 * MolecularNeighborhood::~MolecularNeighborhood
 */
MolecularNeighborhood::~MolecularNeighborhood(void) {
    this->Release();
}

/*
 * MolecularNeighborhood::create
 */
bool MolecularNeighborhood::create(void) {
    return true;
}

/*
 * MolecularNeighborhood::release
 */
void MolecularNeighborhood::release(void) {}

/*
 * MolecularNeighborhood::getData
 */
bool MolecularNeighborhood::getData(core::Call& call) {
    MolecularDataCall* outCall = dynamic_cast<MolecularDataCall*>(&call);
    if (outCall == NULL)
        return false;

    MolecularDataCall* inCall = this->getDataSlot.CallAs<MolecularDataCall>();
    if (inCall == NULL)
        return false;
    inCall->SetCalltime(outCall->Calltime());
    if (!(*inCall)(1))
        return false; // TODO call of getExtent should be enough here?
    if (!(*inCall)(0))
        return false;

    outCall->operator=(*inCall); // deep copy

    // compute the neighborhoods
    if (inCall->DataHash() != lastDataHash || this->neighRadiusParam.IsDirty()) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        findNeighborhoods(*inCall, this->neighRadiusParam.Param<param::FloatParam>()->Value());
        lastDataHash = inCall->DataHash();
        this->neighRadiusParam.ResetDirty();
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
            "Neighborhood search took %f seconds.",
            static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0f);

        if (lastHashSent != outCall->DataHash() + 5) {
            outCall->SetDataHash(outCall->DataHash() + 5);
            lastHashSent = outCall->DataHash();
        } else {
            outCall->SetDataHash(outCall->DataHash() + 10);
            lastHashSent = outCall->DataHash();
        }

    } else {
        lastHashSent = outCall->DataHash();
    }

    outCall->SetNeighborhoodSizes(this->neighborhoodSizes.data());
    outCall->SetNeighborhoods(this->dataPointers.data());

    return true;
}

/*
 * MolecularNeighborhood::getExtent
 */
bool MolecularNeighborhood::getExtent(core::Call& call) {

    MolecularDataCall* outCall = dynamic_cast<MolecularDataCall*>(&call);
    if (outCall == NULL)
        return false;

    MolecularDataCall* inCall = this->getDataSlot.CallAs<MolecularDataCall>();
    if (inCall == NULL)
        return false;
    inCall->SetCalltime(outCall->Calltime());
    if (!(*inCall)(1))
        return false; // TODO call of getExtent should be enough here?

    outCall->operator=(*inCall); // deep copy

    return true;
}

/*
 * MolecularNeighborhood::findNeighborhoods
 */
void MolecularNeighborhood::findNeighborhoods(MolecularDataCall& call, float radius) {
    GridNeighbourFinder<float> finder;
    finder.SetPointData(call.AtomPositions(), call.AtomCount(), call.AccessBoundingBoxes().ObjectSpaceBBox(), radius);
    neighborhood.clear();
    neighborhood.resize(call.AtomCount());
    neighborhoodSizes.clear();
    neighborhoodSizes.resize(call.AtomCount());
    dataPointers.clear();
    dataPointers.resize(call.AtomCount());
    for (unsigned int i = 0; i < call.AtomCount(); i++) {
        finder.FindNeighboursInRange(&call.AtomPositions()[i * 3], radius, neighborhood[i]);
        neighborhoodSizes[i] = static_cast<unsigned int>(neighborhood[i].Count());
        dataPointers[i] = neighborhood[i].PeekElements();
    }
}
