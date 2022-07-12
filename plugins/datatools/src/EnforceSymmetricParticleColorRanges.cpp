/*
 * EnforceSymmetricParticleColorRanges.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "EnforceSymmetricParticleColorRanges.h"
#include <algorithm>
#include <cstdint>

using namespace megamol;


/*
 * datatools::EnforceSymmetricParticleColorRanges::EnforceSymmetricParticleColorRanges
 */
datatools::EnforceSymmetricParticleColorRanges::EnforceSymmetricParticleColorRanges(void)
        : AbstractParticleManipulator("outData", "indata") {
    // intentionally empty
}


/*
 * datatools::EnforceSymmetricParticleColorRanges::~EnforceSymmetricParticleColorRanges
 */
datatools::EnforceSymmetricParticleColorRanges::~EnforceSymmetricParticleColorRanges(void) {
    this->Release();
}


/*
 * datatools::EnforceSymmetricParticleColorRanges::manipulateData
 */
bool datatools::EnforceSymmetricParticleColorRanges::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    outData = inData;                   // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData
    unsigned int plc = outData.GetParticleListCount();
    for (unsigned int i = 0; i < plc; i++) {
        auto& p = outData.AccessParticles(i);
        if (p.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I)
            continue;
        float colIdxVal = std::max<float>(std::abs(p.GetMinColourIndexValue()), std::abs(p.GetMaxColourIndexValue()));
        p.SetColourMapIndexValues(-colIdxVal, colIdxVal);
    }
    return true;
}
