/*
 * IColInverse.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "IColInverse.h"
#include "datatools/MultiParticleDataAdaptor.h"
#include "vislib/math/ShallowVector.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::datatools;


IColInverse::IColInverse()
        : datatools::AbstractParticleManipulator("outData", "inData")
        , dataHash(0)
        , frameID(0)
        , colors()
        , minCol(0.0f)
        , maxCol(1.0f) {
    // intentionally empty
}

IColInverse::~IColInverse() {
    Release();
}

bool IColInverse::manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

    outData = inData;
    inData.SetUnlocker(nullptr, false);

    if ((outData.DataHash() != dataHash) || (outData.DataHash() == 0) || (frameID != outData.FrameID())) {
        dataHash = outData.DataHash();
        frameID = outData.FrameID();

        for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
            auto& plist = outData.AccessParticles(list);
            if (list == 0) {
                minCol = plist.GetMinColourIndexValue();
                maxCol = plist.GetMaxColourIndexValue();
                if (maxCol < minCol)
                    std::swap(minCol, maxCol);
            } else {
                float cMin = plist.GetMinColourIndexValue();
                float cMax = plist.GetMaxColourIndexValue();
                if (cMin > cMax)
                    std::swap(cMin, cMax);
                if (minCol > cMin)
                    minCol = cMin;
                if (maxCol < cMax)
                    maxCol = cMax;
            }
        }

        datatools::MultiParticleDataAdaptor parts(inData);
        colors.resize(parts.get_count());
        for (size_t i = 0; i < parts.get_count(); ++i) {
            colors[i] = minCol + maxCol - parts.get_color(i)[0];
        }
    }

    const float* data = colors.data();
    for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
        auto& plist = outData.AccessParticles(list);
        plist.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I, data, 0);
        plist.SetColourMapIndexValues(minCol, maxCol);
        data += plist.GetCount();
    }

    return true;
}
