/*
 * MpiNode.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "Remote_Service.hpp"

#include "comm/DistributedProto.h"

#ifdef WITH_MPI
    #include <mpi.h>
#else
    using MPI_Comm = int;
    #define MPI_COMM_NULL 0x04000000
#endif

struct megamol::frontend::Remote_Service::MpiNode {
    ~MpiNode();

    bool init(int broadcast_rank);
    bool close();

    bool i_do_broadcast() const { return rank_ == broadcast_rank_; }

    // if broadcast rank: broadcast message to all others
    // if other rank: receive message from broadcast
    bool get_broadcast_message(megamol::remote::Message_t& message);

    void sync_barrier();

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = -1;
    int comm_size_ = 0;
    int broadcast_rank_ = -2;
};

