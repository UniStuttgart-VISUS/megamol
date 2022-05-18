/*
 * IColAdd.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "IColAdd.h"
#include "datatools/MultiParticleDataAdaptor.h"
#include "mmcore/param/FloatParam.h"
#include "stdafx.h"
#include "vislib/math/ShallowVector.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::datatools;


IColAdd::IColAdd()
        : datatools::AbstractParticleManipulator("outData", "inDataA")
        , inDataBSlot("inDataB", "Fetches the second ICol value stream")
        ,
        //aOffsetSlot("aOffset", "Offset to values of stream A"),
        aScaleSlot("aScale", "Scale for values of stream A")
        ,
        //bOffsetSlot("bOffset", "Offset to values of stream B"),
        bScaleSlot("bScale", "Scale to values of stream B")
        , inAHash(0)
        , inBHash(0)
        , outHash(0)
        , frameID(0)
        , colors()
        , minCol(0.0f)
        , maxCol(1.0f) {

    inDataBSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&inDataBSlot);

    //aOffsetSlot.SetParameter(new core::param::FloatParam(0.0f));
    //MakeSlotAvailable(&aOffsetSlot);

    aScaleSlot.SetParameter(new core::param::FloatParam(1.0f));
    MakeSlotAvailable(&aScaleSlot);

    //bOffsetSlot.SetParameter(new core::param::FloatParam(0.0f));
    //MakeSlotAvailable(&bOffsetSlot);

    bScaleSlot.SetParameter(new core::param::FloatParam(1.0f));
    MakeSlotAvailable(&bScaleSlot);
}

IColAdd::~IColAdd() {
    Release();
}

bool IColAdd::manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inDataA) {

    geocalls::MultiParticleDataCall* inDataBptr = inDataBSlot.CallAs<geocalls::MultiParticleDataCall>();
    if (inDataBptr == nullptr)
        return false;
    geocalls::MultiParticleDataCall& inDataB = *inDataBptr;

    inDataB.SetFrameID(inDataA.FrameID(), true);
    if (!inDataB(0))
        return false;

    if ((inAHash != inDataA.DataHash()) || (inDataA.DataHash() == 0) || (inBHash != inDataB.DataHash()) ||
        (inDataB.DataHash() == 0) ||
        (frameID != inDataA.FrameID())
        //            || aOffsetSlot.IsDirty()
        || aScaleSlot.IsDirty()
        //            || bOffsetSlot.IsDirty()
        || bScaleSlot.IsDirty()) {
        // Update data
        inAHash = inDataA.DataHash();
        inBHash = inDataB.DataHash();
        outHash++;
        frameID = inDataA.FrameID();
        //        aOffsetSlot.ResetDirty();
        aScaleSlot.ResetDirty();
        //        bOffsetSlot.ResetDirty();
        bScaleSlot.ResetDirty();

        //        float aOff = aOffsetSlot.Param<core::param::FloatParam>()->Value();
        float aScl = aScaleSlot.Param<core::param::FloatParam>()->Value();
        //        float bOff = bOffsetSlot.Param<core::param::FloatParam>()->Value();
        float bScl = bScaleSlot.Param<core::param::FloatParam>()->Value();

        datatools::MultiParticleDataAdaptor a(inDataA);
        datatools::MultiParticleDataAdaptor b(inDataB);

        if (a.get_count() != b.get_count()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Data streams of A and B are not of same size");
            inDataB.Unlock();
            return false;
        }

        colors.resize(a.get_count());
        if (colors.size() > 0) {
            for (size_t i = 0; i < a.get_count(); ++i) {
                colors[i] = aScl * (*a.get_color(i)) + bScl * (*b.get_color(i));
            }

            minCol = maxCol = colors[0];
            for (size_t i = 1; i < a.get_count(); ++i) {
                if (minCol > colors[i])
                    minCol = colors[i];
                if (maxCol < colors[i])
                    maxCol = colors[i];
            }

        } else {
            minCol = 0.0f;
            maxCol = 1.0f;
        }
    }

    inDataB.Unlock();

    outData = inDataA;
    outData.SetDataHash(outHash);
    outData.SetFrameID(frameID);
    inDataA.SetUnlocker(nullptr, false);

    const float* data = colors.data();
    for (unsigned int list = 0; list < outData.GetParticleListCount(); ++list) {
        auto& plist = outData.AccessParticles(list);
        plist.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I, data, 0);
        plist.SetColourMapIndexValues(minCol, maxCol);
        data += plist.GetCount();
    }

    return true;
}
