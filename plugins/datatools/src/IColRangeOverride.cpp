/*
 * IColRangeOverride.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "IColRangeOverride.h"
#include "datatools/MultiParticleDataAdaptor.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "stdafx.h"
#include "vislib/math/ShallowVector.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::datatools;


IColRangeOverride::IColRangeOverride()
        : datatools::AbstractParticleManipulator("outData", "inDataA")
        , overrideSlot("override", "Enables the value override")
        , inValsSlot("inVals", "Reports the incoming value range. Inputs will be ignored and overwritten.")
        , minValSlot("minVal", "The minimum value for the ICol range output")
        , maxValSlot("maxVal", "The maximum value for the ICol range output")
        , hash(0)
        , frameID(0)
        , minCol(0.0f)
        , maxCol(1.0f) {

    overrideSlot.SetParameter(new core::param::BoolParam(false));
    MakeSlotAvailable(&overrideSlot);

    inValsSlot.SetParameter(new core::param::Vector2fParam(vislib::math::Vector<float, 2>(minCol, maxCol)));
    MakeSlotAvailable(&inValsSlot);

    minValSlot.SetParameter(new core::param::FloatParam(minCol));
    MakeSlotAvailable(&minValSlot);

    maxValSlot.SetParameter(new core::param::FloatParam(maxCol));
    MakeSlotAvailable(&maxValSlot);
}

IColRangeOverride::~IColRangeOverride() {
    Release();
}

bool IColRangeOverride::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

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

        inValsSlot.Param<core::param::Vector2fParam>()->SetValue(vislib::math::Vector<float, 2>(minCol, maxCol), false);
    }

    outData = inData;
    inData.SetUnlocker(nullptr, false);

    if (overrideSlot.Param<core::param::BoolParam>()->Value()) {
        for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
            outData.AccessParticles(list).SetColourMapIndexValues(minValSlot.Param<core::param::FloatParam>()->Value(),
                maxValSlot.Param<core::param::FloatParam>()->Value());
        }
    }

    return true;
}
