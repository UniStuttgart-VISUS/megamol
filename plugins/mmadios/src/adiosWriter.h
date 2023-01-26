#pragma once

#include "mmadios/CallADIOSData.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractDataWriter.h"
#include "vislib/String.h"
#include <adios2.h>
#ifdef MEGAMOL_USE_MPI
#include <mpi.h>
#endif

namespace megamol {
namespace adios {

class adiosWriter : public core::AbstractDataWriter {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "adiosWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data writer module for ADIOS-based IO.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    adiosWriter();

    /** Dtor. */
    ~adiosWriter() override;

    bool create() override;

protected:
    void release() override;

    /**
     * The main function
     *
     * @return True on success
     */
    bool run() override;

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    bool getCapabilities(core::DataWriterCtrlCall& call) override;

private:
    /** slot for MPIprovider */
    core::CallerSlot callRequestMpi;
    bool initMPI();
    vislib::StringA getCommandLine();

#ifdef MEGAMOL_USE_MPI
    MPI_Comm mpi_comm_ = MPI_COMM_NULL;
    bool useMpi = false;
    int mpiRank = -1, mpiSize = -1;
    bool MpiInitialized = false;
#endif

    /** Param Slots */
    core::param::ParamSlot filename;
    core::param::ParamSlot outputPatternSlot;
    core::param::ParamSlot encodingSlot;

    /** The slot asking for data */
    core::CallerSlot getData;

    // ADIOS Stuff
    adios2::ADIOS adiosInst;
    std::shared_ptr<adios2::IO> io;
    adios2::Engine writer;
};


} /* end namespace adios */
} /* end namespace megamol */
