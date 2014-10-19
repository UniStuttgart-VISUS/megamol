/*
 * MpiProvider.cpp
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MpiProvider.h"

#include "MpiCall.h"

#include "vislib/assert.h"
#include "vislib/CmdLineProvider.h"
#include "vislib/Log.h"


/*
 * megamol::core::cluster::mpi::MpiProvider::IsAvailable
 */
bool megamol::core::cluster::mpi::MpiProvider::IsAvailable(void) {
#ifdef WITH_MPI
    return true;
#else /* WITH_MPI */
    return false;
#endif /* WITH_MPI */
}


/*
 * megamol::core::cluster::mpi::MpiProvider::MpiProvider
 */
megamol::core::cluster::mpi::MpiProvider::MpiProvider(void) : Base(),
        callProvideMpi("provideMpi", "Provides the MPI communicator etc.") {
#ifdef WITH_MPI
    this->comm = MPI_COMM_NULL;
#endif /* WITH_MPI */

    this->callProvideMpi.SetCallback(MpiProvider::ClassName(),
        MpiCall::FunctionName(MpiCall::IDX_PROVIDE_MPI),
        &MpiProvider::OnCallProvideMpi);
    this->MakeSlotAvailable(&this->callProvideMpi);
}


/*
 * megamol::core::cluster::mpi::MpiProvider::~MpiProvider
 */
megamol::core::cluster::mpi::MpiProvider::~MpiProvider(void) {
    this->Release();
}


/*
 * megamol::core::cluster::mpi::MpiProvider::create
 */
bool megamol::core::cluster::mpi::MpiProvider::create(void) {
    return true;
}


/*
 * megamol::core::cluster::mpi::MpiProvider::OnCallProvideMpi
 */
bool megamol::core::cluster::mpi::MpiProvider::OnCallProvideMpi(Call& call) {
#ifdef WITH_MPI
    try {
        auto c = dynamic_cast<MpiCall&>(call);
        ASSERT(!c.GetIsInitialising());

        if (this->comm == MPI_COMM_NULL) {
            if (!this->initialiseMpi(c.GetColour())) {
                return false;
            }
            c.SetIsInitialising(true);
        }
        ASSERT(this->comm != MPI_COMM_NULL);

        c.SetComm(this->comm);

        return true;
    } catch (...) {
        return false;
    }

#else /* WITH_MPI */
    return false;
#endif /* WITH_MPI */
}


/*
 * megamol::core::cluster::mpi::MpiProvider::release
 */
void megamol::core::cluster::mpi::MpiProvider::release(void) {
#ifdef WITH_MPI
    if (this->comm != MPI_COMM_NULL) {
        ::MPI_Comm_free(&this->comm);
        this->comm = MPI_COMM_NULL;
        ::MPI_Finalize();
    }
#endif /* WITH_MPI */
}


/*
 * megamol::core::cluster::mpi::MpiProvider::getCommandLine
 */
vislib::StringA megamol::core::cluster::mpi::MpiProvider::getCommandLine(void) {
#ifdef WIN32
    return vislib::StringA(::GetCommandLineA());
#else /* _WIN32 */
    // TODO
#endif /* _WIN32 */
}


/*
 * megamol::core::cluster::mpi::MpiProvider::initialiseMpi
 */
bool megamol::core::cluster::mpi::MpiProvider::initialiseMpi(const int colour) {
    using vislib::sys::Log;

#ifdef WITH_MPI
    int isInitialised = false;
    int rank = 0;

    ::MPI_Initialized(&isInitialised);

    if (isInitialised != 0) {
        Log::DefaultLog.WriteWarn("MPI has already been initialised before "
            "MpiProvider::initialiseMpi was called. This might indicate an "
            "application error as some one else is also using MPI, which we "
            "did not expect.");

    } else {
        vislib::sys::CmdLineProviderA cmdLine(MpiProvider::getCommandLine());
        auto argc = cmdLine.ArgC();
        auto argv = cmdLine.ArgV();

        Log::DefaultLog.WriteInfo("Initialising MPI ...");
        auto status =::MPI_Init(&argc, &argv);
        if (status != MPI_SUCCESS) {
            Log::DefaultLog.WriteError("MPI_Init failed with error code %d",
                status);
            return false;
        }
    }

    Log::DefaultLog.WriteInfo("Performing node colouring with colour %d ...",
        colour);
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &this->comm);
    // TODO: Check status?

    return true;

#else /* WITH_MPI */
    return false;
#endif /* WITH_MPI */
}
