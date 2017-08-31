/*
 * ErosionField.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ErosionField.h"
#include "mmstd_datatools/MultiParticleDataAdaptor.h"
#include "mmstd_datatools/GraphDataCall.h"
#include "vislib/math/ShallowVector.h"
#include <algorithm>
#include "mmcore/param/FloatParam.h"
#include <vector>

using namespace megamol;
using namespace megamol::stdplugin::datatools;


ErosionField::ErosionField() : stdplugin::datatools::AbstractParticleManipulator("outData", "inPtData"),
        inNDataSlot("inNeighborData", "Fetches the neighborhood graph"),
        inPtHash(0), inNHash(0), outHash(0),
        frameID(0), colors(), maxCol(1.0f) {

    inNDataSlot.SetCompatibleCall<stdplugin::datatools::GraphDataCallDescription>();
    MakeSlotAvailable(&inNDataSlot);
}

ErosionField::~ErosionField() {
    Release();
}

bool ErosionField::manipulateData(
        core::moldyn::MultiParticleDataCall& outData,
        core::moldyn::MultiParticleDataCall& inPtData) {

    GraphDataCall *inNDataPtr = inNDataSlot.CallAs<stdplugin::datatools::GraphDataCall>();
    if (inNDataPtr == nullptr) return false;
    GraphDataCall& inNData = *inNDataPtr;

    inNData.SetFrameID(inPtData.FrameID());
    if (!inNData(GraphDataCall::GET_DATA)) return false;

    if ((inPtHash != inPtData.DataHash()) || (inPtData.DataHash() == 0)
            || (inNHash != inNData.DataHash()) || (inNData.DataHash() == 0)
            || (frameID != inPtData.FrameID()) ) {
        // Update data
        inPtHash = inPtData.DataHash();
        inNHash = inNData.DataHash();
        outHash++;
        frameID = inPtData.FrameID();

        stdplugin::datatools::MultiParticleDataAdaptor Pts(inPtData);

        colors.resize(Pts.get_count());
        for (size_t i = 1; i < Pts.get_count(); ++i) {
            if (Pts.get_color(i)[0] < 0.5f) {
                colors[i] = 0.0f;
            } else {
                colors[i] = -1.0f;
            }
        }

        maxCol = 0.0f;
        float nextCol = 0.0f;
        while (std::abs(maxCol - nextCol) < 0.01f) {
            nextCol += 1.0f;

            for (const auto& edge : inNData) {
                float& c1 = colors[edge.i1];
                float& c2 = colors[edge.i2];
                if ((c1 < -0.9f) && (c2 > -0.1f) && (c2 < nextCol - 0.1f)) {
                    c1 = nextCol;
                    maxCol = nextCol;
                } else if ((c2 < -0.9f) && (c1 > -0.1f) && (c1 < nextCol - 0.1f)) {
                    c2 = nextCol;
                    maxCol = nextCol;
                }
            }
        }

    }

    inNData.Unlock();

    outData = inPtData;
    outData.SetDataHash(outHash);
    outData.SetFrameID(frameID);
    inPtData.SetUnlocker(nullptr, false);

    const float *data = colors.data();
    for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
        auto &plist = outData.AccessParticles(list);
        plist.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, data, 0);
        plist.SetColourMapIndexValues(0.0f, maxCol);
        data += plist.GetCount();
    }

    return true;
}
