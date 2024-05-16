/*
 * CopyParticleGlobals.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#include "CopyParticleGlobals.h"

#include <algorithm>
#include <cstdint>

#include "mmcore/param/BoolParam.h"

using namespace megamol;


/*
 * datatools::CopyParticleGlobals::CopyParticleGlobals
 */
datatools::CopyParticleGlobals::CopyParticleGlobals()
        : AbstractParticleManipulator("outData", "indata")
        , inGlobalsSlot("inGlobals", "The particles holding the globals to be copied")
        , copyRadiusSlot("overrideRadius", "Activates copying the radius")
        , copyColorSlot("overrideColor", "Activates copying the color") {

    inGlobalsSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&inGlobalsSlot);

    this->copyRadiusSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->copyRadiusSlot);

    this->copyColorSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->copyColorSlot);
}


/*
 * datatools::CopyParticleGlobals::~CopyParticleGlobals
 */
datatools::CopyParticleGlobals::~CopyParticleGlobals() {
    this->Release();
}


/*
 * datatools::CopyParticleGlobals::manipulateData
 */
bool datatools::CopyParticleGlobals::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* inGlobalsData = inGlobalsSlot.CallAs<geocalls::MultiParticleDataCall>();
    if (inGlobalsData == nullptr)
        return false;
    inGlobalsData->SetFrameID(inData.FrameID(), true);
    if (!(*inGlobalsData)(0))
        return false;

    bool copyRadius = this->copyRadiusSlot.Param<core::param::BoolParam>()->Value();
    bool copyColor = this->copyColorSlot.Param<core::param::BoolParam>()->Value();

    //outData = inData; // also transfers the unlocker to 'outData'

    //inData.SetUnlocker(nullptr, false); // keep original data locked
    // original data will be unlocked through outData

    if (this->AnyParameterDirty()) {
        myHash++;
        this->ResetAllDirtyFlags();
    }
    outData.SetDataHash(myHash);

    unsigned int plc = inData.GetParticleListCount();
    outData.SetParticleListCount(plc);
    for (unsigned int i = 0; i < plc; i++) {
        MultiParticleDataCall::Particles& in_p = inData.AccessParticles(i);
        MultiParticleDataCall::Particles& out_p = outData.AccessParticles(i);
        out_p = in_p;
    }

    if (!copyColor && !copyRadius)
        return true;

    const unsigned int plc_globals = inGlobalsData->GetParticleListCount();
    for (unsigned int i = 0; i < plc; i++) {
        MultiParticleDataCall::Particles& p = outData.AccessParticles(i);
        if (i >= plc_globals) // we ran out of stuff to copy
            continue;
        MultiParticleDataCall::Particles& p_globals = inGlobalsData->AccessParticles(i);

        if (copyRadius) {
            p.SetGlobalRadius(p_globals.GetGlobalRadius());
        }

        if (copyColor) {
            const auto color = p_globals.GetGlobalColour();
            p.SetGlobalColour(color[0], color[1], color[2], color[3]);
            p.SetColourData(MultiParticleDataCall::Particles::COLDATA_NONE, nullptr);
        }
    }

    return true;
}

bool datatools::CopyParticleGlobals::manipulateExtent(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);

    if (this->AnyParameterDirty()) {
        myHash++;
        this->ResetAllDirtyFlags();
    }
    outData.SetDataHash(myHash);

    // we need to adjust the clip box if we fiddle with the radius
    if (this->copyRadiusSlot.Param<core::param::BoolParam>()->Value()) {
        float rad = 0.0f;
        for (auto i = 0; i < outData.GetParticleListCount(); ++i) {
            rad = std::max(rad, outData.AccessParticles(i).GetGlobalRadius());
        }
        auto bbox = outData.AccessBoundingBoxes().ObjectSpaceBBox();
        bbox.Grow(rad);
        outData.AccessBoundingBoxes().SetObjectSpaceClipBox(bbox);
    }

    return true;
}
