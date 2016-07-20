#include "stdafx.h"
#include "RemapIColValues.h"
#include "mmstd_datatools/ParticleFilterMapDataCall.h"

using namespace megamol;
using namespace megamol::stdplugin;
using namespace megamol::stdplugin::datatools;

RemapIColValues::RemapIColValues() : AbstractParticleManipulator("outData", "inData"),
        inIColValuesSlot("inIColData", "The particles holding the ICol data to be used"),
        inParticleMapSlot("inMapData", "The particle index mapping data"),
        dataHash(0), frameId(0), col() {

    inIColValuesSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&inIColValuesSlot);

    inParticleMapSlot.SetCompatibleCall<ParticleFilterMapDataCallDescription>();
    MakeSlotAvailable(&inParticleMapSlot);
}

RemapIColValues::~RemapIColValues() {
    Release();
}

bool RemapIColValues::manipulateData(
        core::moldyn::MultiParticleDataCall& outData,
        core::moldyn::MultiParticleDataCall& inData) {
    core::moldyn::MultiParticleDataCall *inIColData = inIColValuesSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inIColData == nullptr) return false;
    ParticleFilterMapDataCall *inMapData = inParticleMapSlot.CallAs<ParticleFilterMapDataCall>();
    if (inMapData == nullptr) return false;


    // TODO: Implement


    return false;
}
