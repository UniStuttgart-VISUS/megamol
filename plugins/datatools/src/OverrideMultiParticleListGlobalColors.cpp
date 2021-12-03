/*
 * OverrideMultiParticleListGlobalColors.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "OverrideMultiParticleListGlobalColors.h"
#include "stdafx.h"
#include "vislib/graphics/ColourHSVf.h"
#include "vislib/graphics/ColourRGBAu8.h"

using namespace megamol;


datatools::OverrideMultiParticleListGlobalColors::OverrideMultiParticleListGlobalColors(void)
        : AbstractParticleManipulator("outData", "indata") {}

datatools::OverrideMultiParticleListGlobalColors::~OverrideMultiParticleListGlobalColors(void) {
    this->Release();
}

bool datatools::OverrideMultiParticleListGlobalColors::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    outData = inData;                   // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    unsigned int plc = inData.GetParticleListCount();
    for (unsigned int i = 0; i < plc; i++) {
        MultiParticleDataCall::Particles& p = outData.AccessParticles(i);
        vislib::graphics::ColourRGBAu8 rgb(
            vislib::graphics::ColourHSVf(360.0f * static_cast<float>(i) / static_cast<float>(plc), 1.0f, 1.0f));
        p.SetGlobalColour(rgb.R(), rgb.G(), rgb.B());
        p.SetColourData(MultiParticleDataCall::Particles::COLDATA_NONE, nullptr);
    }

    return true;
}
