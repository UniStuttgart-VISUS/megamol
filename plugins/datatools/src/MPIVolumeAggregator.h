/*
 * MPIVolumeAggregator.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractVolumeManipulator.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/param/ParamSlot.h"

#ifdef WITH_MPI
#include "mpi.h"
#endif /* WITH_MPI */

namespace megamol {
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
    static const char* ClassName(void) {
        return "MPIVolumeAggregator";
    }

    /** Return module class description */
    static const char* Description(void) {
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
    bool manipulateData(geocalls::VolumetricDataCall& outData, geocalls::VolumetricDataCall& inData) override;
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

    geocalls::VolumetricDataCall::Metadata metadata;

    int mpiRank = 0;
    int mpiSize = 0;

    std::vector<float> theVolume;
};

} /* end namespace datatools */
} /* end namespace megamol */
