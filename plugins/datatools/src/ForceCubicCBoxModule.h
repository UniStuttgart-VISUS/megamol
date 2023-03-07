/*
 * ForceCubicCBoxModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol::datatools {

/**
 * Module overriding global attributes of particles
 */
class ForceCubicCBoxModule : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ForceCubicCBoxModule";
    }

    /** Return module class description */
    static const char* Description() {
        return "Module forcing the clip box of particles to be cubic";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ForceCubicCBoxModule();

    /** Dtor */
    ~ForceCubicCBoxModule() override;

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

    /**
     * Manipulates the particle data extend information
     *
     * @remarks the default implementation does not changed the data
     *
     * @param outData The call receiving the manipulated information
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    bool manipulateExtent(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
};

} // namespace megamol::datatools
