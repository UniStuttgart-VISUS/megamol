/*
 * ParticleListMergeModule.h
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLELISTMERGEMODULE_H_INCLUDED
#define MEGAMOLCORE_PARTICLELISTMERGEMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "vislib/math/Cuboid.h"
#include "TransferFunctionQuery.h"


namespace megamol {
namespace stdplugin {
namespace datatools {


    /**
     * In-Between management module to change time codes of a data set
     */
    class ParticleListMergeModule : public AbstractParticleManipulator {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ParticleListMergeModule";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module to merge all lists from the core::moldyn::MultiParticleDataCall into a single list";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        ParticleListMergeModule(void);

        /** Dtor. */
        virtual ~ParticleListMergeModule(void);

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

        /**
         * Copies the incoming data 'inDat' into the object's fields
         *
         * @param inDat The incoming data
         */
        void setData(core::moldyn::MultiParticleDataCall& inDat);

        /** The transfer function query */
        TransferFunctionQuery tfq;

        /** The hash id of the data stored */
        size_t dataHash;

        /** The frame id of the data stored */
        unsigned int frameId;

        /** The single list of particles */
        core::moldyn::MultiParticleDataCall::Particles parts;

        /** The stored particle data */
        vislib::RawStorage data;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLELISTMERGEMODULE_H_INCLUDED */
