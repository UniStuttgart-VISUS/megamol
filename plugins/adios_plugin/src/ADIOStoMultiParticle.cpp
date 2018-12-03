/*
 * ADIOStoMultiParticle.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ADIOStoMultiParticle.h"
#include "CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace adios {

ADIOStoMultiParticle::ADIOStoMultiParticle(void)
    : core::Module()
    , mpSlot("mpSlot", "Slot to request multi particle data.")
    , adiosSlot("adiosSlot", "Slot to request ADIOS IO") {

    this->mpSlot.SetCallback("MultiParticleDataCall", "GetData", &ADIOStoMultiParticle::getDataCallback);
    this->mpSlot.SetCallback("MultiParticleDataCall", "GetExtent", &ADIOStoMultiParticle::getExtentCallback);
    this->MakeSlotAvailable(&this->mpSlot);

    this->adiosSlot.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->adiosSlot);
}

ADIOStoMultiParticle::~ADIOStoMultiParticle(void) { this->Release(); }

bool ADIOStoMultiParticle::create(void) { return true; }

void ADIOStoMultiParticle::release(void) {}

bool ADIOStoMultiParticle::getDataCallback(core::Call& call) {
    core::moldyn::MultiParticleDataCall* c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (c2 == nullptr) return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;


    cad->setFrameIDtoLoad(c2->FrameID());

    if (!(*cad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetHeader");
        return false;
    }

    cad->inquire("x");
    cad->inquire("y");
    cad->inquire("z");
    cad->inquire("box");

    if (!(*cad)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetData");
        return false;
    }

    auto X = cad->getData("x")->GetAsFloat();
    auto Y = cad->getData("y")->GetAsFloat();
    auto Z = cad->getData("z")->GetAsFloat();
    auto box = cad->getData("box")->GetAsFloat();

    // Set bounding box
    vislib::math::Cuboid<float> cubo(
        box[0], box[1], box[2], box[3], box[4], box[5]);
    c2->AccessBoundingBoxes().SetObjectSpaceBBox(cubo);
    c2->AccessBoundingBoxes().SetObjectSpaceClipBox(cubo);

    // Set particles
    size_t particleCount = X.size();

    mix.clear();
    mix.resize(particleCount * 3);

    for (int i = 0; i < particleCount; i++) {
         mix[3 * i + 0] = X[i];
         mix[3 * i + 1] = Y[i];
         mix[3 * i + 2] = Z[i];
    }

    c2->SetFrameCount(cad->getFrameCount());
    c2->SetFrameID(0);
    c2->SetDataHash(cad->getDataHash());
    c2->SetParticleListCount(1);
    c2->AccessParticles(0).SetGlobalRadius(1.0f);
    c2->AccessParticles(0).SetCount(particleCount);
    c2->AccessParticles(0).SetGlobalColour(180, 180, 180);



    c2->AccessParticles(0).SetVertexData(
        megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, mix.data(), 3 * sizeof(float));
    c2->AccessParticles(0).SetColourData(
        megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, nullptr);

    return true;
}

bool ADIOStoMultiParticle::getExtentCallback(core::Call& call) {

    core::moldyn::MultiParticleDataCall* mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (mpdc == NULL) return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == NULL) return false;

    this->getDataCallback(call);

    return true;
}

} // end namespace adios
} // end namespace megamol