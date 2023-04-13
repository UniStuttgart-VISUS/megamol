/*
 * TunnelToParticles.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "TunnelToParticles.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "protein_calls/TunnelResidueDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::geocalls;

/*
 * TunnelToParticles::TunnelToParticles
 */
TunnelToParticles::TunnelToParticles()
        : Module()
        , dataOutSlot("getData", "Slot providing the tunnel data as particles")
        , tunnelInSlot("tunnelIn", "Slot taking the tunnel data as input") {

    // caller slot
    this->tunnelInSlot.SetCompatibleCall<protein_calls::TunnelResidueDataCallDescription>();
    this->MakeSlotAvailable(&this->tunnelInSlot);

    // callee slot
    this->dataOutSlot.SetCallback(
        MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0), &TunnelToParticles::getData);
    this->dataOutSlot.SetCallback(
        MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1), &TunnelToParticles::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // parameters
}

/*
 * TunnelToParticles::~TunnelToParticles
 */
TunnelToParticles::~TunnelToParticles() {
    this->Release();
}

/*
 * TunnelToParticles::create
 */
bool TunnelToParticles::create() {
    return true;
}

/*
 * TunnelToParticles::release
 */
void TunnelToParticles::release() {}

/*
 * TunnelToParticles::getData
 */
bool TunnelToParticles::getData(Call& call) {
    MultiParticleDataCall* mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (mpdc == nullptr)
        return false;

    protein_calls::TunnelResidueDataCall* trdc = this->tunnelInSlot.CallAs<protein_calls::TunnelResidueDataCall>();
    if (trdc == nullptr)
        return false;

    if (!(*trdc)(0))
        return false;

    mpdc->SetFrameCount(1); // TODO
    mpdc->SetParticleListCount(trdc->getTunnelNumber());

    MultiParticleDataCall::Particles::VertexDataType vrtDatType = MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR;
    MultiParticleDataCall::Particles::ColourDataType colDatType = MultiParticleDataCall::Particles::COLDATA_NONE;
    unsigned int stride = 16;

    for (int i = 0; i < trdc->getTunnelNumber(); i++) {
        MultiParticleDataCall::Particles& pts = mpdc->AccessParticles(i);
        pts.SetGlobalColour(0, 0, 255);
        pts.SetCount(trdc->getTunnelDescriptions()[i].coordinates.size());
        pts.SetVertexData(vrtDatType, trdc->getTunnelDescriptions()[i].coordinates.data(), stride);
        pts.SetColourData(colDatType, trdc->getTunnelDescriptions()[i].coordinates.data(), stride);
    }

    return true;
}

/*
 * TunnelToParticles::getExtent
 */
bool TunnelToParticles::getExtent(Call& call) {
    MultiParticleDataCall* mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (mpdc == nullptr)
        return false;

    protein_calls::TunnelResidueDataCall* trdc = this->tunnelInSlot.CallAs<protein_calls::TunnelResidueDataCall>();
    if (trdc == nullptr)
        return false;

    if (!(*trdc)(1))
        return false;
    if (!(*trdc)(0))
        return false;
    // the call to getData here is necessary because the loader cannot
    // know the number of tunnels before loading the whole file

    mpdc->SetFrameCount(1); // TODO
    mpdc->SetParticleListCount(trdc->getTunnelNumber());
    mpdc->AccessBoundingBoxes().Clear();
    mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(trdc->GetBoundingBoxes().ObjectSpaceBBox());
    mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(trdc->GetBoundingBoxes().ObjectSpaceClipBox());

    return true;
}
