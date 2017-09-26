/*
 * ParticleListSelector.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLELISTSELECTOR_H_INCLUDED
#define MEGAMOLCORE_PARTICLELISTSELECTOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module thinning the number of particles
     *
     * Migrated from SGrottel particle's tool box
     */
    class ParticleListSelector : public AbstractParticleManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "ParticleListSelector";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "Selects a single list of particles from a MultiParticleDataCall";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        ParticleListSelector(void);

        /** Dtor */
        virtual ~ParticleListSelector(void);

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

        /** The list selection index */
        core::param::ParamSlot listIndexSlot;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLELISTSELECTOR_H_INCLUDED */
