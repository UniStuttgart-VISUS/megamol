/*
 * ParticleIColGradientField.h
 *
 * Copyright (C) 2016 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLEICOLGRADIENTFIELD_H_INCLUDED
#define MEGAMOLCORE_PARTICLEICOLGRADIENTFIELD_H_INCLUDED
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>


namespace megamol {
namespace datatools {

/**
 * Module overriding global attributes of particles
 */
class ParticleIColGradientField : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleIColGradientField";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Computes the gradient field on particles with IColor, and stores the corresponding vector fiels as "
               "RGBf color data";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    ParticleIColGradientField(void);

    /** Dtor */
    virtual ~ParticleIColGradientField(void);

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
    virtual bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

private:
    void compute_colors(geocalls::MultiParticleDataCall& dat);
    void set_colors(geocalls::MultiParticleDataCall& dat);

    core::param::ParamSlot radiusSlot;
    size_t datahash;
    unsigned int time;
    std::vector<float> newColors;
    float maxColor;
};

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLEICOLGRADIENTFIELD_H_INCLUDED */
