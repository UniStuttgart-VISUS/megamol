/*
 * MPIParticleCollector.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MPIPARTICLECOLLECTOR_H_INCLUDED
#define MEGAMOLCORE_MPIPARTICLECOLLECTOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"

#ifdef WITH_MPI
#include "mpi.h"
#endif /* WITH_MPI */

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module merging object-space distributed MultiparticleDataCalls over MPI.
     * This should be used for gathering large in situ SUBSAMPLED (ParticleThinner) data sets:
     * Everything is collected at once and MPI cannot push that much data
     * at once.
     */
    class MPIParticleCollector : public AbstractParticleManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "MPIParticleCollector";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "merges object-space distributed MultiparticleDataCalls over MPI";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
#ifdef WITH_MPI
            return true;
#else
            return false;
#endif
        }

        /** Ctor */
        MPIParticleCollector(void);

        /** Dtor */
        virtual ~MPIParticleCollector(void);

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
        bool initMPI();

    private:

#ifdef WITH_MPI
        /** The communicator that the view uses. */
        MPI_Comm comm = MPI_COMM_NULL;
#endif /* WITH_MPI */

        /** slot for MPIprovider */
        core::CallerSlot callRequestMpi;

        int mpiRank = 0;
        int mpiSize = 0;

        std::vector<uint8_t> vertexData, colorData;
        std::vector<uint8_t> allVertexData, allColorData;
    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MPIPARTICLECOLLECTOR_H_INCLUDED */
