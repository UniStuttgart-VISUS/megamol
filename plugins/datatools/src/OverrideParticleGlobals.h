/*
 * OverrideParticleGlobals.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_OVERRIDEPARTICLEGLOBALS_H_INCLUDED
#define MEGAMOLCORE_OVERRIDEPARTICLEGLOBALS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"
#include <limits>

namespace megamol {
namespace datatools {

/**
 * Module overriding global attributes of particles
 */
class OverrideParticleGlobals : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "OverrideParticleGlobals";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Module overriding global attributes of particles";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    OverrideParticleGlobals(void);

    /** Dtor */
    virtual ~OverrideParticleGlobals(void);

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
    bool anythingDirty();

    void resetAllDirty();

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

    /** the new range */
    core::param::ParamSlot minIntSlot;
    core::param::ParamSlot maxIntSlot;

    SIZE_T myHash = std::numeric_limits<SIZE_T>::max();
};

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_OVERRIDEPARTICLEGLOBALS_H_INCLUDED */
