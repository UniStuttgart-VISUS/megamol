/*
 * EnforceSymmetricParticleColorRanges.h
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
class EnforceSymmetricParticleColorRanges : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "EnforceSymmetricParticleColorRanges";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Changes all color index ranges to be symmetric around zero.";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    EnforceSymmetricParticleColorRanges(void);

    /** Dtor */
    ~EnforceSymmetricParticleColorRanges(void) override;

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
};

} /* end namespace datatools */
} /* end namespace megamol */
