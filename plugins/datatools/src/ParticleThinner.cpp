/*
 * ParticleThinner.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "ParticleThinner.h"
#include "mmcore/param/IntParam.h"
#include "stdafx.h"

using namespace megamol;


/*
 * datatools::ParticleThinner::ParticleThinner
 */
datatools::ParticleThinner::ParticleThinner(void)
        : AbstractParticleManipulator("outData", "indata")
        , thinningFactorSlot("thinningFactor", "The thinning factor. Only each n-th particle will be kept.") {
    this->thinningFactorSlot.SetParameter(new core::param::IntParam(100, 1));
    this->MakeSlotAvailable(&this->thinningFactorSlot);
}


/*
 * datatools::ParticleThinner::~ParticleThinner
 */
datatools::ParticleThinner::~ParticleThinner(void) {
    this->Release();
}


/*
 * datatools::ParticleThinner::manipulateData
 */
bool datatools::ParticleThinner::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;
    int tf = this->thinningFactorSlot.Param<core::param::IntParam>()->Value();

    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    unsigned int plc = outData.GetParticleListCount();
    for (unsigned int i = 0; i < plc; i++) {
        MultiParticleDataCall::Particles& p = outData.AccessParticles(i);

        UINT64 cnt = p.GetCount();

        const void* cd = p.GetColourData();
        unsigned int cds = p.GetColourDataStride();
        MultiParticleDataCall::Particles::ColourDataType cdt = p.GetColourDataType();

        const void* vd = p.GetVertexData();
        unsigned int vds = p.GetVertexDataStride();
        MultiParticleDataCall::Particles::VertexDataType vdt = p.GetVertexDataType();

        cds *= tf; // lol
        vds *= tf;
        cnt /= tf;

        p.SetCount(cnt);
        p.SetColourData(cdt, cd, cds);
        p.SetVertexData(vdt, vd, vds);
    }

    return true;
}
