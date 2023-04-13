/*
 * OverrideParticleGlobals.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"
#include <limits>

namespace megamol::datatools {

/**
 * Module overriding global attributes of particles
 */
class OverrideParticleGlobals : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "OverrideParticleGlobals";
    }

    /** Return module class description */
    static const char* Description() {
        return "Module overriding global attributes of particles";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    OverrideParticleGlobals();

    /** Dtor */
    ~OverrideParticleGlobals() override;

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
    /** Activates overriding the selected values for all particle lists */
    core::param::ParamSlot overrideAllListSlot;

    /** The particle list to override the values of */
    core::param::ParamSlot overrideListSlot;

    /** Activates overriding the radius */
    core::param::ParamSlot overrideRadiusSlot;

    /** Activates overriding the color */
    core::param::ParamSlot overrideColorSlot;

    /** Activates overriding the intensity range */
    core::param::ParamSlot overrideIntensityRangeSlot;

    /** The new color value */
    core::param::ParamSlot colorSlot;

    /** The new radius */
    core::param::ParamSlot radiusSlot;

    bool anythingDirty();

    void resetAllDirty();

    SIZE_T myHash = std::numeric_limits<SIZE_T>::max();

    /** the new range */
    core::param::ParamSlot minIntSlot;
    core::param::ParamSlot maxIntSlot;
};

} // namespace megamol::datatools
