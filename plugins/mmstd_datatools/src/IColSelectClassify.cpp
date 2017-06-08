/*
 * IColSelectClassify.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "IColSelectClassify.h"
#include "mmstd_datatools/MultiParticleDataAdaptor.h"
#include <algorithm>
#include "mmcore/param/FloatParam.h"

using namespace megamol;
using namespace megamol::stdplugin::datatools;


IColSelectClassify::IColSelectClassify() : stdplugin::datatools::AbstractParticleManipulator("outData", "inData"),
        valueSlot("value", "The selected value of the input ICol data"),
        epsilonSlot("epsilon", "The (input) ICol value comparison epsilon"),
        inHash(0), outHash(0), frameID(0), colors() {

    valueSlot.SetParameter(new core::param::FloatParam(1.0f));
    MakeSlotAvailable(&valueSlot);

    epsilonSlot.SetParameter(new core::param::FloatParam(0.001f, 0.0f));
    MakeSlotAvailable(&epsilonSlot);

}

IColSelectClassify::~IColSelectClassify() {
    Release();
}

bool IColSelectClassify::manipulateData(
        core::moldyn::MultiParticleDataCall& outData,
        core::moldyn::MultiParticleDataCall& inData) {

    if ( (inHash != inData.DataHash()) || (inData.DataHash() == 0)
            || (frameID != inData.FrameID())
            || valueSlot.IsDirty()
            || epsilonSlot.IsDirty()) {
        // Update data
        inHash = inData.DataHash();
        outHash++;
        frameID = inData.FrameID();
        valueSlot.ResetDirty();
        epsilonSlot.ResetDirty();

        float e = epsilonSlot.Param<core::param::FloatParam>()->Value();
        float v1 = valueSlot.Param<core::param::FloatParam>()->Value();
        float v2 = v1 + e;
        v1 -= e;

        stdplugin::datatools::MultiParticleDataAdaptor d(inData);

        colors.resize(d.get_count());
        for (size_t i = 0; i < d.get_count(); ++i) {
            float v = d.get_color(i)[0];
            colors[i] = ((v > v1) && (v < v2)) ? 1.0f : 0.0f;
        }
    }

    outData = inData;
    outData.SetDataHash(outHash);
    outData.SetFrameID(frameID);
    inData.SetUnlocker(nullptr, false);

    const float *data = colors.data();
    for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
        auto &plist = outData.AccessParticles(list);
        plist.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, data, 0);
        plist.SetColourMapIndexValues(0.0f, 1.0f);
        data += plist.GetCount();
    }

    return true;
}
