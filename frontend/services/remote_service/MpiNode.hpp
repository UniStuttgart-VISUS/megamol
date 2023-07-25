/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "Remote_Service.hpp"

#include "comm/DistributedProto.h"

#include "MPI_Context.h"

struct megamol::frontend::Remote_Service::MpiNode {
    ~MpiNode();

    bool init(int broadcast_rank);
    bool close();

    // if broadcast rank: broadcast message to all others
    // if other rank: receive message from broadcast
    bool get_broadcast_message(megamol::remote::Message_t& message);

    bool i_do_broadcast() {
        return mpi_comm.do_i_broadcast();
    }

    void sync_barrier();

    frontend_resources::MPI_Context mpi_comm;
};
