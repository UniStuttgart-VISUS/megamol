/*
 * ModColIRange.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "ModColIRange.h"
#include "datatools/MultiParticleDataAdaptor.h"
#include "mmcore/param/FloatParam.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::datatools;


ModColIRange::ModColIRange()
        : datatools::AbstractParticleManipulator("outData", "inData")
        , rangeSlot("maxVal", "Specifies the color range from [0..r[")
        , inDataHash(0)
        , outDataHash(0)
        , frameID(0)
        , colors()
        , minCol(0)
        , maxCol(1) {

    rangeSlot.SetParameter(new core::param::FloatParam(1.0f, 1.0f));
    MakeSlotAvailable(&rangeSlot);
}

ModColIRange::~ModColIRange() {
    Release();
}

bool ModColIRange::manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

    if ((inData.DataHash() != inDataHash) || (inData.FrameID() != frameID) || (inData.DataHash() == 0) ||
        rangeSlot.IsDirty()) {
        inDataHash = inData.DataHash();
        outDataHash++;
        frameID = inData.FrameID();
        rangeSlot.ResetDirty();

        datatools::MultiParticleDataAdaptor parts(inData);
        if (parts.get_count() > 0) {

            colors.resize(parts.get_count());
            minCol = 0.0f;
            maxCol = rangeSlot.Param<core::param::FloatParam>()->Value();
            for (size_t i = 0; i < parts.get_count(); ++i) {
                float c = *parts.get_color(i);
                unsigned long f = static_cast<unsigned long>(c / maxCol);
                c -= static_cast<float>(f) * maxCol;
                colors[i] = c;
            }

        } else {
            minCol = 0.0f;
            maxCol = 1.0f;
        }
    }

    outData = inData;
    inData.SetUnlocker(nullptr, false);
    outData.SetFrameID(frameID);
    outData.SetDataHash(outDataHash);
    const float* data = colors.data();
    for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
        auto& plist = outData.AccessParticles(list);
        plist.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I, data, 0);
        data += plist.GetCount();
        plist.SetColourMapIndexValues(minCol, maxCol);
    }

    return true;
}
