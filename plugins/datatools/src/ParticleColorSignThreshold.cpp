/*
 * ParticleColorSignThreshold.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "ParticleColorSignThreshold.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include <algorithm>
#include <cstdint>

using namespace megamol;


/*
 * datatools::ParticleColorSignThreshold::ParticleColorSignThreshold
 */
datatools::ParticleColorSignThreshold::ParticleColorSignThreshold(void)
        : AbstractParticleManipulator("outData", "indata")
        , enableSlot("enable", "Enables the color manipulation")
        , negativeThresholdSlot("negativeThreshold", "Color values below this threshold will be mapped to -1")
        , positiveThresholdSlot("positiveThreshold", "Color values above this threshold will be mapped to 1")
        , datahash(0)
        , time(0)
        , newColors() {

    this->enableSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->enableSlot);

    this->negativeThresholdSlot.SetParameter(new core::param::FloatParam(-0.01f));
    this->MakeSlotAvailable(&this->negativeThresholdSlot);

    this->positiveThresholdSlot.SetParameter(new core::param::FloatParam(0.01f));
    this->MakeSlotAvailable(&this->positiveThresholdSlot);
}


/*
 * datatools::ParticleColorSignThreshold::~ParticleColorSignThreshold
 */
datatools::ParticleColorSignThreshold::~ParticleColorSignThreshold(void) {
    this->Release();
}


/*
 * datatools::ParticleColorSignThreshold::manipulateData
 */
bool datatools::ParticleColorSignThreshold::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    outData = inData;                   // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    if (!this->enableSlot.Param<core::param::BoolParam>()->Value())
        return true;

    if (this->negativeThresholdSlot.IsDirty()) {
        this->negativeThresholdSlot.ResetDirty();
        this->datahash = 0;
    }
    if (this->positiveThresholdSlot.IsDirty()) {
        this->positiveThresholdSlot.ResetDirty();
        this->datahash = 0;
    }
    if ((this->datahash == 0) || (this->datahash != outData.DataHash()) || (this->time != outData.FrameID())) {
        this->datahash = outData.DataHash();
        this->time = outData.FrameID();
        this->compute_colors(outData);
    }

    if (this->newColors.size() > 0) {
        this->set_colors(outData);
    }

    return true;
}


void datatools::ParticleColorSignThreshold::compute_colors(geocalls::MultiParticleDataCall& dat) {
    size_t allpartcnt = 0;

    unsigned int plc = dat.GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I)
            continue;
        allpartcnt += static_cast<size_t>(pl.GetCount());
    }

    this->newColors.resize(allpartcnt);

    float negcol = this->negativeThresholdSlot.Param<core::param::FloatParam>()->Value();
    float poscol = this->positiveThresholdSlot.Param<core::param::FloatParam>()->Value();

    allpartcnt = 0;
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I)
            continue;

        int part_cnt = static_cast<int>(pl.GetCount());
        const unsigned char* col = static_cast<const unsigned char*>(pl.GetColourData());
        unsigned int stride = std::max<unsigned int>(pl.GetColourDataStride(), sizeof(float));
#pragma omp parallel for
        for (int part_i = 0; part_i < part_cnt; ++part_i) {
            float c = *reinterpret_cast<const float*>(col + (part_i * stride));
            if (c < negcol)
                c = -1.0f;
            else if (c > poscol)
                c = 1.0f;
            else
                c = 0.0f;
            this->newColors[allpartcnt + part_i] = c;
        }

        allpartcnt += static_cast<size_t>(part_cnt);
    }
}


void datatools::ParticleColorSignThreshold::set_colors(geocalls::MultiParticleDataCall& dat) {
    size_t allpartcnt = 0;

    unsigned int plc = dat.GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I)
            continue;

        pl.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I, this->newColors.data() + allpartcnt);
        pl.SetColourMapIndexValues(-1.0f, 1.0f);

        allpartcnt += static_cast<size_t>(pl.GetCount());
    }
}
