/*
 * ParticleInstantiator.h
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLEINSTANTIATOR_H_INCLUDED
#define MEGAMOLCORE_PARTICLEINSTANTIATOR_H_INCLUDED
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace datatools {

/**
 * Module overriding global colors of multi particle lists
 */
class ParticleInstantiator : public AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "ParticleInstantiator";
    }
    static const char* Description() {
        return "makes instances of particles by xyz repetition";
    }
    static bool IsAvailable() {
        return true;
    }

    ParticleInstantiator();
    ~ParticleInstantiator() override;

protected:
    bool InterfaceIsDirty();
    void InterfaceResetDirty();

    /**
     * Manipulates the particle data
     *
     * @remarks the default implementation does not changed the data
     *
     * @param outData The call receiving the manipulated data
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

    bool manipulateExtent(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    megamol::core::param::ParamSlot numInstancesParam;
    megamol::core::param::ParamSlot instanceOffsetParam;

    megamol::core::param::ParamSlot setFromClipboxParam;
    megamol::core::param::ParamSlot setFromBoundingboxParam;

    SIZE_T in_hash = -1;
    unsigned int in_frameID = -1;
    SIZE_T my_hash = 0;
    std::vector<std::vector<float>> vertData;
    std::vector<std::vector<uint8_t>> colData;
    std::vector<std::vector<float>> dirData;
    std::vector<bool> has_global_radius;
};

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLEINSTANTIATOR_H_INCLUDED */
