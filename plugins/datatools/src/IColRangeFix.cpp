/*
 * IColRangeFix.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "IColRangeFix.h"
#include "datatools/MultiParticleDataAdaptor.h"
#include "mmcore/param/FloatParam.h"
#include "stdafx.h"
#include "vislib/math/ShallowVector.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::datatools;


IColRangeFix::IColRangeFix()
        : datatools::AbstractParticleManipulator("outData", "inDataA")
        , hash(0)
        , frameID(0)
        , minCol(0.0f)
        , maxCol(1.0f) {
    // intentionally empty
}

IColRangeFix::~IColRangeFix() {
    Release();
}

bool IColRangeFix::manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

    if ((hash != inData.DataHash()) || (inData.DataHash() == 0) || (frameID != inData.FrameID())) {
        // Update data
        hash = inData.DataHash();
        frameID = inData.FrameID();

        datatools::MultiParticleDataAdaptor d(inData);

        if (d.get_count() > 0) {
            minCol = maxCol = *d.get_color(0);
            for (size_t i = 1; i < d.get_count(); ++i) {
                float c = *d.get_color(i);
                if (minCol > c)
                    minCol = c;
                if (maxCol < c)
                    maxCol = c;
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
