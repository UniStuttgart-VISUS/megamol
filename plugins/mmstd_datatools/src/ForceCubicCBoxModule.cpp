/*
 * ForceCubicCBoxModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ForceCubicCBoxModule.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ForceCubicCBoxModule::ForceCubicCBoxModule
 */
datatools::ForceCubicCBoxModule::ForceCubicCBoxModule(void)
        : AbstractParticleManipulator("outData", "indata") {
    // intentionally empty
}


/*
 * datatools::ForceCubicCBoxModule::~ForceCubicCBoxModule
 */
datatools::ForceCubicCBoxModule::~ForceCubicCBoxModule(void) {
    this->Release();
}


/*
 * datatools::ForceCubicCBoxModule::manipulateData
 */
bool datatools::ForceCubicCBoxModule::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    if (outData.AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
        float leh = 0.5f * outData.AccessBoundingBoxes().ObjectSpaceClipBox().LongestEdge();
        vislib::math::Point<float, 3> pt = outData.AccessBoundingBoxes().ObjectSpaceClipBox().CalcCenter();
        outData.AccessBoundingBoxes().SetObjectSpaceClipBox(pt.X() - leh, pt.Y() - leh, pt.Z() - leh, pt.X() + leh, pt.Y() + leh, pt.Z() + leh);
    }

    //if (outData.AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
    //    float leh = 0.5f * outData.AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    //    vislib::math::Point<float, 3> pt = outData.AccessBoundingBoxes().ObjectSpaceBBox().CalcCenter();
    //    outData.AccessBoundingBoxes().SetObjectSpaceBBox(pt.X() - leh, pt.Y() - leh, pt.Z() - leh, pt.X() + leh, pt.Y() + leh, pt.Z() + leh);
    //}

    return true;
}


/*
 * datatools::ForceCubicCBoxModule::manipulateExtent
 */
bool datatools::ForceCubicCBoxModule::manipulateExtent(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    if (outData.AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
        float leh = 0.5f * outData.AccessBoundingBoxes().ObjectSpaceClipBox().LongestEdge();
        vislib::math::Point<float, 3> pt = outData.AccessBoundingBoxes().ObjectSpaceClipBox().CalcCenter();
        outData.AccessBoundingBoxes().SetObjectSpaceClipBox(pt.X() - leh, pt.Y() - leh, pt.Z() - leh, pt.X() + leh, pt.Y() + leh, pt.Z() + leh);
    }

    //if (outData.AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
    //    float leh = 0.5f * outData.AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    //    vislib::math::Point<float, 3> pt = outData.AccessBoundingBoxes().ObjectSpaceBBox().CalcCenter();
    //    outData.AccessBoundingBoxes().SetObjectSpaceBBox(pt.X() - leh, pt.Y() - leh, pt.Z() - leh, pt.X() + leh, pt.Y() + leh, pt.Z() + leh);
    //}

    return true;
}
