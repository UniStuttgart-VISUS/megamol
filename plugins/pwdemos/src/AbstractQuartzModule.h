/*
 * AbstractQuartzModule.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CallerSlot.h"
#include "QuartzCrystalDataCall.h"
#include "QuartzParticleGridDataCall.h"


namespace megamol {
namespace demos {

    /**
     * Abstract base class for quartzs data consuming modules
     */
    class AbstractQuartzModule {
    public:

        /**
         * Ctor
         */
        AbstractQuartzModule(void);

        /**
         * Dtor
         */
        virtual ~AbstractQuartzModule(void);

    protected:

        /**
         * Answer the particle data from the connected module
         *
         * @return The particle data from the connected module or NULL if no
         *        data could be received
         */
        ParticleGridDataCall *getParticleData(void);

        /**
         * Answer the crystalite data from the connected module
         *
         * @return The crystalite data from the connected module or NULL if no
         *         data could be received
         */
        virtual CrystalDataCall *getCrystaliteData(void);

        /** The slot to get the data */
        core::CallerSlot dataInSlot;

        /** The slot to get the types data */
        core::CallerSlot typesInSlot;

        /** The data hash of the types data realized the last time */
        SIZE_T typesDataHash;

    };

} /* end namespace demos */
} /* end namespace megamol */

