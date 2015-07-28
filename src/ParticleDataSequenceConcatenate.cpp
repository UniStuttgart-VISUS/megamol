#include "stdafx.h"
#include "ParticleDataSequenceConcatenate.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

using namespace megamol;
using namespace megamol::stdplugin::datatools;

ParticleDataSequenceConcatenate::ParticleDataSequenceConcatenate()
        : Module(),
        dataOutSlot("out", "Publishes the concatenated data"),
        dataIn1Slot("in1", "First data source"),
        dataIn2Slot("in2", "Second data source") {

    dataOutSlot.SetCallback("MultiParticleDataCall", "GetData", &ParticleDataSequenceConcatenate::getData);
    dataOutSlot.SetCallback("MultiParticleDataCall", "GetExtent", &ParticleDataSequenceConcatenate::getExtend);
    MakeSlotAvailable(&dataOutSlot);

    dataIn1Slot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&dataIn1Slot);

    dataIn2Slot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
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
    // intentionally empty
    return false;
}

bool ParticleDataSequenceConcatenate::getData(megamol::core::Call& c) {
    // intentionally empty
    return false;
}
