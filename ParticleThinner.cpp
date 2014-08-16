/*
 * ParticleThinner.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleThinner.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleThinner::ParticleThinner
 */
datatools::ParticleThinner::ParticleThinner(void)
        : AbstractParticleManipulator() {
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
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;
    outData = inData;

    unsigned int plc = outData.GetParticleListCount();
    for (unsigned int i = 0; i < plc; i++) {
        MultiParticleDataCall::Particles& p = outData.AccessParticles(i);

        UINT64 cnt = p.GetCount();

        const void *cd = p.GetColourData();
        unsigned int cds = p.GetColourDataStride();
        MultiParticleDataCall::Particles::ColourDataType cdt = p.GetColourDataType();

        const void *vd = p.GetVertexData();
        unsigned int vds = p.GetVertexDataStride();
        MultiParticleDataCall::Particles::VertexDataType vdt = p.GetVertexDataType();

        cds *= 10;
        vds *= 10;
        cnt /= 10;

        p.SetCount(cnt);
        p.SetColourData(cdt, cd, cds);
        p.SetVertexData(vdt, vd, vds);
    }

    return true;
}
