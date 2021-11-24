/*
 * MPDCListsConcatenate.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "MPDCListsConcatenate.h"
#include "stdafx.h"

#include "geometry_calls/MultiParticleDataCall.h"


megamol::datatools::MPDCListsConcatenate::MPDCListsConcatenate()
        : dataOutSlot("out", "Publishes the concatenated data")
        , dataIn1Slot("in1", "First data source")
        , dataIn2Slot("in2", "Second data source") {
    dataOutSlot.SetCallback("MultiParticleDataCall", "GetData", &MPDCListsConcatenate::getData);
    dataOutSlot.SetCallback("MultiParticleDataCall", "GetExtent", &MPDCListsConcatenate::getExtent);
    MakeSlotAvailable(&dataOutSlot);

    dataIn1Slot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&dataIn1Slot);

    dataIn2Slot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&dataIn2Slot);
}


megamol::datatools::MPDCListsConcatenate::~MPDCListsConcatenate() {
    this->Release();
}


bool megamol::datatools::MPDCListsConcatenate::create(void) {
    return true;
}


void megamol::datatools::MPDCListsConcatenate::release(void) {}


bool megamol::datatools::MPDCListsConcatenate::getExtent(megamol::core::Call& c) {
    using geocalls::MultiParticleDataCall;
    auto oc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (oc == nullptr)
        return false;
    auto i1c = dataIn1Slot.CallAs<MultiParticleDataCall>();
    auto i2c = dataIn2Slot.CallAs<MultiParticleDataCall>();

    // test if only at most one of the in calls is connected
    if (i1c == nullptr) {
        if (i2c != nullptr) {
            *i2c = *oc;
            if (!((*i2c)(1)))
                return false;
            *oc = *i2c;
            i2c->SetUnlocker(nullptr, false);
            return true;
        } else
            return false;
    } else if (i2c == nullptr) {
        *i1c = *oc;
        if (!((*i1c)(1)))
            return false;
        *oc = *i1c;
        i1c->SetUnlocker(nullptr, false);
        return true;
    }

    // both calls are connected, so be smart!
    if (!((*i1c)(1)))
        return false;
    if (!((*i2c)(1)))
        return false;

    auto const i1fc = i1c->FrameCount();
    auto const i2fc = i2c->FrameCount();

    auto const minFc = std::min(i1fc, i2fc);

    auto reqFid = oc->FrameID();

    if (reqFid >= minFc)
        reqFid = minFc - 1;

    i1c->SetFrameID(reqFid, oc->IsFrameForced());
    if (!((*i1c)(1)))
        return false;

    i2c->SetFrameID(reqFid, oc->IsFrameForced());
    if (!((*i2c)(1)))
        return false;

    vislib::math::Cuboid<float> osbb(i1c->AccessBoundingBoxes().ObjectSpaceBBox());
    osbb.Union(i2c->AccessBoundingBoxes().ObjectSpaceBBox());

    oc->SetFrameCount(minFc);
    oc->SetFrameID(reqFid, oc->IsFrameForced());

    oc->AccessBoundingBoxes().SetObjectSpaceBBox(osbb);
    oc->SetDataHash(i1c->DataHash() + i2c->DataHash() * 10);

    return true;
}


bool megamol::datatools::MPDCListsConcatenate::getData(megamol::core::Call& c) {
    using geocalls::MultiParticleDataCall;
    auto oc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (oc == nullptr)
        return false;
    auto i1c = dataIn1Slot.CallAs<MultiParticleDataCall>();
    auto i2c = dataIn2Slot.CallAs<MultiParticleDataCall>();

    // test if only at most one of the in calls is connected
    if (i1c == nullptr) {
        if (i2c != nullptr) {
            *i2c = *oc;
            if (!((*i2c)(0)))
                return false;
            *oc = *i2c;
            i2c->SetUnlocker(nullptr, false);
            return true;
        } else
            return false;
    } else if (i2c == nullptr) {
        *i1c = *oc;
        if (!((*i1c)(0)))
            return false;
        *oc = *i1c;
        i1c->SetUnlocker(nullptr, false);
        return true;
    }

    // both calls are connected, so be smart!

    if (!((*i1c)(0)))
        return false;
    if (!((*i2c)(0)))
        return false;

    auto const i1plc = i1c->GetParticleListCount();
    auto const i2plc = i2c->GetParticleListCount();

    auto const plc = i1plc + i2plc;
    oc->SetParticleListCount(plc);

    for (unsigned int plidx = 0; plidx < i1plc; ++plidx) {
        auto& outPl = oc->AccessParticles(plidx);
        auto const& inPl = i1c->AccessParticles(plidx);
        outPl = inPl;
    }

    for (unsigned int plidx = i1plc; plidx < plc; ++plidx) {
        auto& outPl = oc->AccessParticles(plidx);
        auto const& inPl = i2c->AccessParticles(plidx - i1plc);
        outPl = inPl;
    }

    oc->SetDataHash(i1c->DataHash() + i2c->DataHash() * 10);

    return true;
}
