/*
 * MPI_Context.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#ifdef MEGAMOL_USE_MPI
#include <mpi.h>
#else
using MPI_Comm = int;
#define MPI_COMM_NULL 0x04000000
#endif

namespace megamol::frontend_resources {

struct MPI_Context {
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    int mpi_comm_size = -1;
    int rank = 0;            // rank of this process
    int broadcast_rank = -1; // rank of process that is supposed to do MPI broadcasts

    bool do_i_broadcast() {
        return rank == broadcast_rank;
    };
};

} // namespace megamol::frontend_resources
