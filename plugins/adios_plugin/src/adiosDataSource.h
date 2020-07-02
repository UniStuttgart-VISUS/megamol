#pragma once

#include <adios2.h>
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AnimDataModule.h"
#include "vislib/math/Cuboid.h"
#include "adios_plugin/CallADIOSData.h"
#include "vislib/String.h"
#ifdef WITH_MPI
#    include <mpi.h>
#endif

namespace megamol {
namespace adios {

class adiosDataSource : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "adiosDataSource"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source module for ADIOS-based IO."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    adiosDataSource(void);

    /** Dtor. */
    virtual ~adiosDataSource(void);

    bool create(void);

protected:
    void release(void);

    /**
     * Loads inquired data.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Get meta data.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getHeaderCallback(core::Call& caller);

private:
    /** slot for MPIprovider */
    core::CallerSlot callRequestMpi;
    bool initMPI();

#ifdef WITH_MPI
    MPI_Comm mpi_comm_ = MPI_COMM_NULL;
    int mpiRank = -1, mpiSize = -1;
    bool MpiInitialized = false;
#endif

    vislib::StringA getCommandLine(void);
    bool filenameChanged(core::param::ParamSlot& slot);

    /** The slot for requesting data */
    core::CalleeSlot getData;

    /** The frame index table */
    //std::vector<UINT64> frameIdx;

    /** Data file load id counter */
    size_t data_hash = 0;
	bool dataHashChanged = false;
    bool inquireChanged = false;

    /** The file name */
    core::param::ParamSlot filenameSlot;

    size_t frameCount = 0;
    long long int loadedFrameID = -1;

    // ADIOS Stuff
    std::shared_ptr<adios2::ADIOS> adiosInst;
    std::shared_ptr<adios2::IO> io;
    std::shared_ptr<adios2::Engine> reader;
    std::map<std::string, adios2::Params> variables;
    adiosDataMap dataMap;

    std::vector<std::size_t> timesteps;
    std::vector<std::string> availVars;
};
} /* end namespace adios */
} /* end namespace megamol */
