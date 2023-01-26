/*
 * ParticleColorSignThreshold.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLECOLORSIGNTHRESHOLD_H_INCLUDED
#define MEGAMOLCORE_PARTICLECOLORSIGNTHRESHOLD_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>


namespace megamol {
namespace datatools {

/**
 * Module overriding global attributes of particles
 */
class ParticleColorSignThreshold : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleColorSignThreshold";
    }

    /** Return module class description */
    static const char* Description() {
        return "Performs a sign threshold adjustment of the particles' colors";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleColorSignThreshold();

    /** Dtor */
    ~ParticleColorSignThreshold() override;

protected:
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

private:
    void compute_colors(geocalls::MultiParticleDataCall& dat);
    void set_colors(geocalls::MultiParticleDataCall& dat);

    core::param::ParamSlot enableSlot;
    core::param::ParamSlot negativeThresholdSlot;
    core::param::ParamSlot positiveThresholdSlot;
    size_t datahash;
    unsigned int time;
    std::vector<float> newColors;
};

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLECOLORSIGNTHRESHOLD_H_INCLUDED */
