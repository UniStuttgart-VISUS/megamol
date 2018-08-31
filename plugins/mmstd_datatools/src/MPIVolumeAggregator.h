/*
 * MPIVolumeAggregator.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MPIVOLUMEAGGREGATOR_H_INCLUDED
#define MEGAMOLCORE_MPIVOLUMEAGGREGATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmstd_datatools/AbstractVolumeManipulator.h"
#include "mmcore/param/ParamSlot.h"

#ifdef WITH_MPI
#include "mpi.h"
#endif /* WITH_MPI */

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module aggregating the density of several identically-sized volumes over MPI.
     * This should be used for gathering large in situ SUBSAMPLED (ParticleThinner) data sets:
     * Everything is collected at once and MPI cannot push that much data
     * at once.
     */
    class MPIVolumeAggregator : public AbstractVolumeManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "MPIVolumeAggregator";
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
        MPIVolumeAggregator(void);

        /** Dtor */
        virtual ~MPIVolumeAggregator(void);

    protected:

        /**
         * Manipulates the volume data
         *
         * @remarks the default implementation does not changed the data
         *
         * @param outData The call receiving the manipulated data
         * @param inData The call holding the original data
         *
         * @return True on success
         */
        bool manipulateData (
            megamol::core::misc::VolumetricDataCall& outData, megamol::core::misc::VolumetricDataCall& inData) override;
        bool initMPI();

        void release(void) override;

    private:

#ifdef WITH_MPI
        /** The communicator that the view uses. */
        MPI_Comm comm = MPI_COMM_NULL;
#endif /* WITH_MPI */

        /** slot for MPIprovider */
        core::CallerSlot callRequestMpi;

        core::param::ParamSlot operatorSlot;

        core::misc::VolumetricDataCall::Metadata metadata;

        int mpiRank = 0;
        int mpiSize = 0;

        std::vector<float> theVolume;
    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MPIVOLUMEAGGREGATOR_H_INCLUDED */
