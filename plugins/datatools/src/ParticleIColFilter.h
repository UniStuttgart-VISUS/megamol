/*
 * ParticleIColFilter.h
 *
 * Copyright (C) 2015-2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_PARTICLEICOLFILTER_H_INCLUDED
#define MEGAMOLCORE_PARTICLEICOLFILTER_H_INCLUDED
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "datatools/ParticleFilterMapDataCall.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"
#include <vector>


namespace megamol {
namespace datatools {


// TODO: make operators available: larger, smaller, between, epsilon-equal

/**
 * Removes particles outside a specific interval for I color values
 */
class ParticleIColFilter : public AbstractParticleManipulator {
public:
    static const char* ClassName(void) {
        return "ParticleIColFilter";
    }
    static const char* Description(void) {
        return "Removes particles outside a specific interval for I color values";
    }
    static bool IsAvailable(void) {
        return true;
    }

    ParticleIColFilter(void);
    ~ParticleIColFilter(void) override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    bool reset(core::param::ParamSlot&);
    void setData(geocalls::MultiParticleDataCall& inDat);
    void setData(geocalls::MultiParticleDataCall::Particles& p, vislib::RawStorage& d,
        const geocalls::SimpleSphericalParticles& s, vislib::math::Cuboid<float> bbox,
        ParticleFilterMapDataCall::index_t& mapOffset);

    bool getParticleMapData(core::Call& c);
    bool getParticleMapExtent(core::Call& c);
    bool getParticleMapHash(core::Call& c);

    bool isDirty() {
        return minValSlot.IsDirty() || maxValSlot.IsDirty();
    }

    void resetDirty() {
        minValSlot.ResetDirty();
        maxValSlot.ResetDirty();
    }

    core::CalleeSlot particleMapSlot;
    core::param::ParamSlot minValSlot;
    core::param::ParamSlot maxValSlot;
    core::param::ParamSlot staifHackDistSlot;
    size_t dataHash;
    size_t outDataHash = 0;
    unsigned int frameId;
    std::vector<geocalls::MultiParticleDataCall::Particles> parts;
    std::vector<vislib::RawStorage> data;
    std::vector<ParticleFilterMapDataCall::index_t> mapIndex;
    core::param::ParamSlot inValRangeSlot;
};

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLEICOLFILTER_H_INCLUDED */
