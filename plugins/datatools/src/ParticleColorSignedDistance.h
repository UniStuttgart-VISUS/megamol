/*
 * ParticleColorSignedDistance.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>


namespace megamol::datatools {

/**
 * Module overriding global attributes of particles
 */
class ParticleColorSignedDistance : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleColorSignedDistance";
    }

    /** Return module class description */
    static const char* Description() {
        return "Computes signed distances of particles to the closest particle with a color value of zero. The sign is "
               "preserved from the original color.";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleColorSignedDistance();

    /** Dtor */
    ~ParticleColorSignedDistance() override;

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
    core::param::ParamSlot cyclXSlot;
    core::param::ParamSlot cyclYSlot;
    core::param::ParamSlot cyclZSlot;
    size_t datahash;
    unsigned int time;
    std::vector<float> newColors;
    float minCol, maxCol;
};

} // namespace megamol::datatools
