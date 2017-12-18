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

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module overriding global attributes of particles
     */
    class OverrideParticleGlobals : public AbstractParticleManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "OverrideParticleGlobals";
        }

        /** Return module class description */
        static const char *Description(void) {
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
        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

        /** Activates overriding the selected values for all particle lists */
        core::param::ParamSlot overrideAllListSlot;

        /** The particle list to override the values of */
        core::param::ParamSlot overrideListSlot;

        /** Activates overriding the radius */
        core::param::ParamSlot overrideRadiusSlot;

        /** The new radius value */
        core::param::ParamSlot radiusSlot;

        /** Activates overriding the color */
        core::param::ParamSlot overrideColorSlot;

        /** The new color value */
        core::param::ParamSlot colorSlot;

        /** Activates overriding the intensity range */
        core::param::ParamSlot overrideIntensityRangeSlot;

        /** the new range */
        core::param::ParamSlot minIntSlot;
        core::param::ParamSlot maxIntSlot;
    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_OVERRIDEPARTICLEGLOBALS_H_INCLUDED */
