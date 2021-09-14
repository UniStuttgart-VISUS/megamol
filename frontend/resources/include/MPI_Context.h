/*
 * MPI_Context.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

namespace megamol {
namespace frontend_resources {

struct MPI_Context {
    int mpi_comm        = 0; // MPI_Comm
    int mpi_comm_size   =-1;
    int rank            = 0; // rank of this process
    int broadcast_rank  =-1; // rank of process that is supposed to do MPI broadcasts

    bool do_i_broadcast() { return rank == broadcast_rank; };
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
