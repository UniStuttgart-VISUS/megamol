/*
 * ForceCubicCBoxModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace datatools {

/**
 * Module overriding global attributes of particles
 */
class ForceCubicCBoxModule : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ForceCubicCBoxModule";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Module forcing the clip box of particles to be cubic";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    ForceCubicCBoxModule(void);

    /** Dtor */
    virtual ~ForceCubicCBoxModule(void);

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
    virtual bool manipulateExtent(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

private:
};

} /* end namespace datatools */
} /* end namespace megamol */
