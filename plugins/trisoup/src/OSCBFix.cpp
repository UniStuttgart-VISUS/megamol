/*
 * OSCBFix.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "OSCBFix.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include <climits>


using namespace megamol;
using namespace megamol::quartz;


/*
 * OSCBFix::OSCBFix
 */
OSCBFix::OSCBFix(void)
        : core::Module()
        , dataOutSlot("dataOut", "The slot providing the fixed data")
        , dataInSlot("dataIn", "The slot fetching the original data")
        , datahash(0)
        , frameNum(UINT_MAX)
        , oscb() {

    this->dataInSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &OSCBFix::getData);
    this->dataOutSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &OSCBFix::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);
}


/*
 * OSCBFix::~OSCBFix
 */
OSCBFix::~OSCBFix(void) {
    this->Release();
}


/*
 * OSCBFix::create
 */
bool OSCBFix::create(void) {
    // intentionally empty
    return true;
}


/*
 * OSCBFix::getData
 */
bool OSCBFix::getData(core::Call& c) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* outCall = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outCall == NULL)
        return false;

    MultiParticleDataCall* inCall = this->dataInSlot.CallAs<MultiParticleDataCall>();
    if (inCall == NULL)
        return false;

    *inCall = *outCall;

    if (!(*inCall)(0))
        return false;

    *outCall = *inCall;

    inCall->SetUnlocker(NULL, false);

    if (outCall->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
        if ((outCall->DataHash() == 0) || (outCall->DataHash() != this->datahash) ||
            (outCall->FrameID() != this->frameNum)) {
            this->frameNum = outCall->FrameID();
            this->datahash = outCall->DataHash();
            this->calcOSCB(*inCall);
        }
        outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->oscb);
    }

    return true;
}


/*
 * OSCBFix::getExtent
 */
bool OSCBFix::getExtent(core::Call& c) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* outCall = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outCall == NULL)
        return false;

    MultiParticleDataCall* inCall = this->dataInSlot.CallAs<MultiParticleDataCall>();
    if (inCall == NULL)
        return false;

    *inCall = *outCall;

    if (!(*inCall)(1))
        return false;

    *outCall = *inCall;

    inCall->SetUnlocker(NULL, false);

    if (outCall->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
        if ((outCall->DataHash() == 0) || (outCall->DataHash() != this->datahash) ||
            (outCall->FrameID() != this->frameNum)) {
            if ((*inCall)(0)) { // we need data!
                this->frameNum = outCall->FrameID();
                this->datahash = outCall->DataHash();
                this->calcOSCB(*inCall);
                inCall->Unlock();
            }
        }
        outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->oscb);
    }

    return true;
}


/*
 * OSCBFix::release
 */
void OSCBFix::release(void) {
    //this->crystals.Clear();
}


/*
 * OSCBFix::calcOSCB
 */
void OSCBFix::calcOSCB(geocalls::MultiParticleDataCall& data) {
    using geocalls::MultiParticleDataCall;
    this->oscb.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    vislib::math::Cuboid<float> box;
    bool first = true;

    for (unsigned int i = 0; i < data.GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles& parts = data.AccessParticles(i);
        MultiParticleDataCall::Particles::VertexDataType vdt = parts.GetVertexDataType();
        const float* vdf = static_cast<const float*>(parts.GetVertexData());
        const short* vds = static_cast<const short*>(parts.GetVertexData());
        unsigned int step = parts.GetVertexDataStride();
        float rad = parts.GetGlobalRadius();
        bool isFloat = true;

        switch (vdt) {
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            step = vislib::math::Max<unsigned int>(step, sizeof(float) * 3);
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            step = vislib::math::Max<unsigned int>(step, sizeof(float) * 4);
            break;
        case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: // quantized, but I ignore this for now
            step = vislib::math::Max<unsigned int>(step, sizeof(short) * 3);
            break;
        default:
            continue;
        }

        for (UINT64 j = 0; j < parts.GetCount(); j++) {
            switch (vdt) {
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                rad = vdf[3];
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                box.Set(vdf[0] - rad, vdf[1] - rad, vdf[2] - rad, vdf[0] + rad, vdf[1] + rad, vdf[2] + rad);
                vdf = reinterpret_cast<const float*>(reinterpret_cast<const char*>(vdf) + step);
                break;
            case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                box.Set(static_cast<float>(vds[0]) - rad, static_cast<float>(vds[1]) - rad,
                    static_cast<float>(vds[2]) - rad, static_cast<float>(vds[0]) + rad,
                    static_cast<float>(vds[1]) + rad, static_cast<float>(vds[2]) + rad);
                vds = reinterpret_cast<const short*>(reinterpret_cast<const char*>(vds) + step);
                break;
            default:
                continue;
            }
            if (!first) {
                this->oscb.Union(box);
            } else {
                this->oscb = box;
                first = false;
            }
        }
    }
}
