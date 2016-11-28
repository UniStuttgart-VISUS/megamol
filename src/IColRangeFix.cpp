/*
 * IColRangeFix.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "IColRangeFix.h"
#include "mmstd_datatools/MultiParticleDataAdaptor.h"
#include "vislib/math/ShallowVector.h"
#include <algorithm>
#include "mmcore/param/FloatParam.h"

using namespace megamol;
using namespace megamol::stdplugin::datatools;


IColRangeFix::IColRangeFix() : stdplugin::datatools::AbstractParticleManipulator("outData", "inDataA"),
        hash(0), frameID(0), minCol(0.0f), maxCol(1.0f) {
    // intentionally empty
}

IColRangeFix::~IColRangeFix() {
    Release();
}

bool IColRangeFix::manipulateData(
        core::moldyn::MultiParticleDataCall& outData,
        core::moldyn::MultiParticleDataCall& inData) {

    if ( (hash != inData.DataHash()) || (inData.DataHash() == 0)
            || (frameID != inData.FrameID()) ) {
        // Update data
        hash = inData.DataHash();
        frameID = inData.FrameID();

        stdplugin::datatools::MultiParticleDataAdaptor d(inData);

        if (d.get_count() > 0) {
            minCol = maxCol = *d.get_color(0);
            for (size_t i = 1; i < d.get_count(); ++i) {
                float c = *d.get_color(i);
                if (minCol > c) minCol = c;
                if (maxCol < c) maxCol = c;
            }

        } else {
            minCol = 0.0f;
            maxCol = 1.0f;
        }

    }

    outData = inData;
    inData.SetUnlocker(nullptr, false);

    for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
        outData.AccessParticles(list).SetColourMapIndexValues(minCol, maxCol);
    }

    return true;
}
