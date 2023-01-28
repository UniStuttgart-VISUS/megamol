/*
 * ParticleThinner.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLETHINNER_H_INCLUDED
#define MEGAMOLCORE_PARTICLETHINNER_H_INCLUDED
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol::datatools {

/**
 * Module thinning the number of particles
 *
 * Migrated from SGrottel particle's tool box
 */
class ParticleThinner : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleThinner";
    }

    /** Return module class description */
    static const char* Description() {
        return "Thins number of particles";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleThinner();

    /** Dtor */
    ~ParticleThinner() override;

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
    /** The thinning factor. Only each n-th particle will be kept. */
    core::param::ParamSlot thinningFactorSlot;
};

} // namespace megamol::datatools

#endif /* MEGAMOLCORE_PARTICLETHINNER_H_INCLUDED */
