/*
 * MPIParticleCollector.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"

#ifdef MEGAMOL_USE_MPI
#include "mpi.h"
#endif /* MEGAMOL_USE_MPI */

namespace megamol {
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
    static const char* ClassName(void) {
        return "MPIParticleCollector";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "merges object-space distributed MultiparticleDataCalls over MPI";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
#ifdef MEGAMOL_USE_MPI
        return true;
#else
        return false;
#endif
    }

    /** Ctor */
    MPIParticleCollector(void);

    /** Dtor */
    ~MPIParticleCollector(void) override;

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
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;
    bool initMPI();

private:
#ifdef MEGAMOL_USE_MPI
    /** The communicator that the view uses. */
    MPI_Comm comm = MPI_COMM_NULL;
#endif /* MEGAMOL_USE_MPI */

    /** slot for MPIprovider */
    core::CallerSlot callRequestMpi;

    int mpiRank = 0;
    int mpiSize = 0;

    std::vector<uint8_t> vertexData, colorData;
    std::vector<uint8_t> allVertexData, allColorData;
};

} /* end namespace datatools */
} /* end namespace megamol */
