/*
 * ParticleListSelector.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol::datatools {

/**
 * Module thinning the number of particles
 *
 * Migrated from SGrottel particle's tool box
 */
class ParticleListSelector : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleListSelector";
    }

    /** Return module class description */
    static const char* Description() {
        return "Selects a single list of particles from a MultiParticleDataCall";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleListSelector();

    /** Dtor */
    ~ParticleListSelector() override;

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
    /** The list selection index */
    core::param::ParamSlot listIndexSlot;
};

} // namespace megamol::datatools
