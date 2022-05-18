#include "ParticleDataSequenceConcatenate.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::datatools;

ParticleDataSequenceConcatenate::ParticleDataSequenceConcatenate()
        : Module()
        , dataOutSlot("out", "Publishes the concatenated data")
        , dataIn1Slot("in1", "First data source")
        , dataIn2Slot("in2", "Second data source") {

    dataOutSlot.SetCallback("MultiParticleDataCall", "GetData", &ParticleDataSequenceConcatenate::getData);
    dataOutSlot.SetCallback("MultiParticleDataCall", "GetExtent", &ParticleDataSequenceConcatenate::getExtend);
    MakeSlotAvailable(&dataOutSlot);

    dataIn1Slot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&dataIn1Slot);

    dataIn2Slot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&dataIn2Slot);
}

ParticleDataSequenceConcatenate::~ParticleDataSequenceConcatenate() {
    this->Release();
}

bool ParticleDataSequenceConcatenate::create(void) {
    // intentionally empty
    return true;
}

void ParticleDataSequenceConcatenate::release(void) {
    // intentionally empty
}

bool ParticleDataSequenceConcatenate::getExtend(megamol::core::Call& c) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* oc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (oc == nullptr)
        return false;
    MultiParticleDataCall* i1c = dataIn1Slot.CallAs<MultiParticleDataCall>();
    MultiParticleDataCall* i2c = dataIn2Slot.CallAs<MultiParticleDataCall>();

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
    unsigned int i1fc = i1c->FrameCount();

    if (oc->FrameID() >= i1fc) {
        i2c->SetFrameID(oc->FrameID() - i1fc, oc->IsFrameForced());
        if (!((*i2c)(1)))
            return false;
    } else {
        i1c->Unlock();
        i1c->SetFrameID(oc->FrameID(), oc->IsFrameForced());
        if (!((*i1c)(1)))
            return false;
        i2c->SetFrameID(0);
        if (!((*i2c)(1)))
            return false;
    }

    vislib::math::Cuboid<float> osbb(i1c->AccessBoundingBoxes().ObjectSpaceBBox());
    osbb.Union(i2c->AccessBoundingBoxes().ObjectSpaceBBox());

    if (oc->FrameID() >= i1fc) {
        *oc = *i2c;
        oc->SetFrameID(oc->FrameID() + i1fc, oc->IsFrameForced());
        i2c->SetUnlocker(nullptr, false);
        oc->SetFrameCount(oc->FrameCount() + i1c->FrameCount());
    } else {
        *oc = *i1c;
        i1c->SetUnlocker(nullptr, false);
        oc->SetFrameCount(oc->FrameCount() + i2c->FrameCount());
    }

    oc->AccessBoundingBoxes().SetObjectSpaceBBox(osbb);
    oc->SetDataHash(i1c->DataHash() + i2c->DataHash() * 10);

    if (oc->FrameID() >= i1fc) {
        i1c->Unlock();
    } else {
        i2c->Unlock();
    }

    return true;
}

bool ParticleDataSequenceConcatenate::getData(megamol::core::Call& c) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* oc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (oc == nullptr)
        return false;
    MultiParticleDataCall* i1c = dataIn1Slot.CallAs<MultiParticleDataCall>();
    MultiParticleDataCall* i2c = dataIn2Slot.CallAs<MultiParticleDataCall>();

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
    if (!((*i1c)(1)))
        return false;
    unsigned int i1fc = i1c->FrameCount();
    size_t i1dh = i1c->DataHash();
    i1c->Unlock();
    if (!((*i2c)(1)))
        return false;
    size_t i2dh = i2c->DataHash();
    i2c->Unlock();

    if (oc->FrameID() < i1fc) {
        // ask i1
        *i1c = *oc;
        if (!((*i1c)(0)))
            return false;
        *oc = *i1c;
        i1c->SetUnlocker(nullptr, false);

    } else {
        // ask i2
        *i2c = *oc;
        i2c->SetFrameID(oc->FrameID() - i1fc, oc->IsFrameForced());
        if (!((*i2c)(0)))
            return false;
        *oc = *i2c;
        oc->SetFrameID(i2c->FrameID() + i1fc, i2c->IsFrameForced());
        i2c->SetUnlocker(nullptr, false);
    }

    oc->SetDataHash(i1dh + i2dh * 10);
    return true;
}
