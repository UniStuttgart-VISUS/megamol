/*
 * CopyParticleGlobals.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <limits>

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::datatools {

/**
 * Module overriding global attributes of particles
 */
class CopyParticleGlobals : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "CopyParticleGlobals";
    }

    /** Return module class description */
    static const char* Description() {
        return "Module copying global attributes of one particle data source into another one as far as possible "
               "(stops after list exhaustion on either side)";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    CopyParticleGlobals();

    /** Dtor */
    ~CopyParticleGlobals() override;

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

    bool manipulateExtent(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    core::CallerSlot inGlobalsSlot;

    /** Activates overriding the radius */
    core::param::ParamSlot copyRadiusSlot;

    /** Activates overriding the color */
    core::param::ParamSlot copyColorSlot;

    SIZE_T myHash = std::numeric_limits<SIZE_T>::max();
};

} // namespace megamol::datatools
