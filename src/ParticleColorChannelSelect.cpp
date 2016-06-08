/*
 * ParticleColorChannelSelect.h
 *
 * Copyright (C) 2016 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleColorChannelSelect.h"
#include "mmcore/param/EnumParam.h"
#include <cstdint>
#include <algorithm>
#include <cassert>

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleColorChannelSelect::ParticleColorChannelSelect
 */
datatools::ParticleColorChannelSelect::ParticleColorChannelSelect(void)
        : AbstractParticleManipulator("outData", "indata"),
        channelSlot("channel", "The color channel to be selected as new I color channel"),
        dataHash(-1), colRange() {

    core::param::EnumParam *chan = new core::param::EnumParam(3);
    chan->SetTypePair(0, "R");
    chan->SetTypePair(1, "G");
    chan->SetTypePair(2, "B");
    chan->SetTypePair(3, "A");
    channelSlot.SetParameter(chan);
    MakeSlotAvailable(&channelSlot);
}


/*
 * datatools::ParticleColorChannelSelect::~ParticleColorChannelSelect
 */
datatools::ParticleColorChannelSelect::~ParticleColorChannelSelect(void) {
    this->Release();
}


/*
 * datatools::ParticleColorChannelSelect::manipulateData
 */
bool datatools::ParticleColorChannelSelect::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    outData = inData; // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData
    int chan = channelSlot.Param<core::param::EnumParam>()->Value();
    if (chan < 0) chan = 0;
    if (chan > 3) chan = 3;

    if (dataHash != inData.DataHash()) {
        dataHash = inData.DataHash();
        colRange.clear();
    }

    unsigned int plc = outData.GetParticleListCount();
    for (unsigned int i = 0; i < plc; i++) {
        auto& p = outData.AccessParticles(i);
        if (p.GetCount() <= 0) continue;

        if (((p.GetColourDataType() == megamol::core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB) && (chan != 3)) 
                || (p.GetColourDataType() == megamol::core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA)) {
            unsigned int stride;
            if (p.GetColourDataType() == megamol::core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB) {
                stride = 3 * 4;
            } else {
                assert(p.GetColourDataType() == megamol::core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA);
                stride = 4 * 4;
            }
            if (p.GetColourDataStride() > stride) stride = p.GetColourDataStride();
            p.SetColourData(megamol::core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I,
                static_cast<const float*>(p.GetColourData()) + chan, stride);

            std::map<const void*, std::pair<float, float> >::iterator clRng = colRange.find(p.GetColourData());

            if (clRng == colRange.end()) {
                // find min and max color values
                const uint8_t *colPtr = static_cast<const uint8_t*>(p.GetColourData()) + sizeof(float) * chan;
                float minV, maxV;
                minV = maxV = *reinterpret_cast<const float*>(colPtr);
                for (uint64_t i = 1; i < p.GetCount(); ++i, colPtr += stride) {
                    const float& f = *reinterpret_cast<const float*>(colPtr);
                    if (minV > f) minV = f;
                    if (maxV < f) maxV = f;
                }

                if (minV > maxV - 0.0001f) {
                    maxV = minV + 0.5f;
                    minV -= 0.5f;
                }

                colRange[p.GetColourData()] = std::pair<float, float>(minV, maxV);
                p.SetColourMapIndexValues(minV, maxV);
            } else {
                p.SetColourMapIndexValues(clRng->second.first, clRng->second.second);
            }
        } // else everything stays as it is
    }
    return true;
}
