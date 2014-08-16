/*
 * ParticleRelaxationModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLERELAXATIONMODULE_H_INCLUDED
#define MEGAMOLCORE_PARTICLERELAXATIONMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractParticleManipulator.h"
#include "param/ParamSlot.h"
#include "TransferFunctionQuery.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module overriding global attributes of particles
     */
    class ParticleRelaxationModule : public AbstractParticleManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "ParticleRelaxationModule";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "Module relaxing particles to minimize overlapps";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        ParticleRelaxationModule(void);

        /** Dtor */
        virtual ~ParticleRelaxationModule(void);

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

        /** The transfer function query */
        TransferFunctionQuery tfq;

        /** The hash id of the data stored */
        size_t dataHash;

        /** The frame id of the data stored */
        unsigned int frameId;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLERELAXATIONMODULE_H_INCLUDED */
