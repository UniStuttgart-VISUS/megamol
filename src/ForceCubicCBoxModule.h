/*
 * ForceCubicCBoxModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FORCECUBICCBOXMODULE_H_INCLUDED
#define MEGAMOLCORE_FORCECUBICCBOXMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module overriding global attributes of particles
     */
    class ForceCubicCBoxModule : public AbstractParticleManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "ForceCubicCBoxModule";
        }

        /** Return module class description */
        static const char *Description(void) {
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
        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

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
        virtual bool manipulateExtent(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FORCECUBICCBOXMODULE_H_INCLUDED */
