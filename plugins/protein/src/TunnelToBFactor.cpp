/*
 * TunnelToBFactor.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "TunnelToBFactor.h"

#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/TunnelResidueDataCall.h"
#include <climits>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 * TunnelToBFactor::TunnelToBFactor
 */
TunnelToBFactor::TunnelToBFactor(void)
        : Module()
        , dataOutSlot("dataOut", "Output slot for the output molecular data")
        , molInSlot("moleculeIn", "Input slot for the molecular data")
        , tunnelInSlot("tunnelIn", "Input slot for the tunnel data") {

    // caller slots
    this->molInSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molInSlot);

    this->tunnelInSlot.SetCompatibleCall<TunnelResidueDataCallDescription>();
    this->MakeSlotAvailable(&this->tunnelInSlot);

    // callee slot
    this->dataOutSlot.SetCallback(
        MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(0), &TunnelToBFactor::getData);
    this->dataOutSlot.SetCallback(
        MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(1), &TunnelToBFactor::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // parameters
}

/*
 * TunnelToBFactor::~TunnelToBFactor
 */
TunnelToBFactor::~TunnelToBFactor(void) {
    this->Release();
}

/*
 * TunnelToBFactor::create
 */
bool TunnelToBFactor::create(void) {
    return true;
}

/*
 * TunnelToBFactor::release
 */
void TunnelToBFactor::release(void) {}

/*
 * TunnelToBFactor::getData
 */
bool TunnelToBFactor::getData(Call& call) {
    MolecularDataCall* outCall = dynamic_cast<MolecularDataCall*>(&call);
    if (outCall == nullptr)
        return false;

    MolecularDataCall* mdc = this->molInSlot.CallAs<MolecularDataCall>();
    if (mdc == nullptr)
        return false;
    TunnelResidueDataCall* trdc = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
    if (trdc == nullptr)
        return false;

    mdc->SetCalltime(outCall->Calltime());
    if (!(*mdc)(0))
        return false;

    if (!(*trdc)(0))
        return false;

    outCall->operator=(*mdc);

    applyBFactor(outCall, mdc, trdc);

    return true;
}

/*
 * TunnelToBFactor::getExtent
 */
bool TunnelToBFactor::getExtent(Call& call) {
    MolecularDataCall* outCall = dynamic_cast<MolecularDataCall*>(&call);
    if (outCall == nullptr)
        return false;

    MolecularDataCall* mdc = this->molInSlot.CallAs<MolecularDataCall>();
    if (mdc == nullptr)
        return false;

    TunnelResidueDataCall* trdc = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
    if (trdc == nullptr)
        return false;

    mdc->SetCalltime(outCall->Calltime());
    if (!(*mdc)(1))
        return false;

    if (!(*trdc)(1))
        return false;

    outCall->operator=(*mdc); // deep copy

    return true;
}

/*
 * TunnelToBFactor::applyBFactor
 */
void TunnelToBFactor::applyBFactor(
    MolecularDataCall* outCall, MolecularDataCall* inCall, TunnelResidueDataCall* tunnelCall) {
    auto numFactors = inCall->AtomCount();
    this->bFactors.resize(numFactors, 0.0f);

    // setup search array
    std::vector<int> revMap;
    int mymin = INT_MAX, mymax = INT_MIN;
    for (int i = 0; i < static_cast<int>(numFactors); i++) {
        if (inCall->AtomFormerIndices()[i] > mymax)
            mymax = inCall->AtomFormerIndices()[i];
        if (inCall->AtomFormerIndices()[i] < mymin)
            mymin = inCall->AtomFormerIndices()[i];
    }

    revMap.resize(mymax + 1);
    for (int i = 0; i < static_cast<int>(numFactors); i++) {
        revMap[inCall->AtomFormerIndices()[i]] = i;
    }

    for (int i = 0; i < tunnelCall->getTunnelNumber(); i++) {
        auto numAtoms = tunnelCall->getTunnelDescriptions()[i].atomIdentifiers.size();
        for (int j = 0; j < static_cast<int>(numAtoms); j++) {
            int atomIdx = tunnelCall->getTunnelDescriptions()[i].atomIdentifiers[j].first;
            this->bFactors[revMap[atomIdx]] = 1.0f;
        }
    }

    outCall->SetAtomBFactors(this->bFactors.data());
    outCall->SetBFactorRange(0.0f, 1.0f);
}
