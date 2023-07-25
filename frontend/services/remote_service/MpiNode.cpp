/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "MpiNode.hpp"

#include <chrono>

#include "mmcore/utility/log/Log.h"
#include "vislib/sys/SystemInformation.h"

#include "comm/DistributedProto.h"
using namespace megamol::remote;

static const std::string service_name = "Remote_Service::MpiNode: ";
static void log(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(msg.c_str());
}

megamol::frontend::Remote_Service::MpiNode::~MpiNode() {
    close();
}

#ifndef MEGAMOL_USE_MPI
bool megamol::frontend::Remote_Service::MpiNode::init(int broadcast_rank) {
    return false;
}
bool megamol::frontend::Remote_Service::MpiNode::close() {
    return false;
}
bool megamol::frontend::Remote_Service::MpiNode::get_broadcast_message(megamol::remote::Message_t& message) {
    return false;
}
void megamol::frontend::Remote_Service::MpiNode::sync_barrier() {}
#else

bool megamol::frontend::Remote_Service::MpiNode::init(int broadcast_rank) {
    int isInitialised = 0;
    MPI_Initialized(&isInitialised);

    auto& comm = mpi_comm.mpi_comm;
    auto& comm_size = mpi_comm.mpi_comm_size;
    auto& rank = mpi_comm.rank;

    if (isInitialised) {
        log_warning("MPI has already been initialised");
        return true;
    }
    if (comm != MPI_COMM_NULL) {
        log_error("MPI has not been initialised but my communicator is not MPI_COMM_NULL");
        return true;
    }
    log("Initialising MPI");

    mpi_comm.broadcast_rank = broadcast_rank;
    comm = MPI_COMM_NULL;
    rank = 0;

    auto init_status = MPI_Init(NULL, NULL);
    if (init_status != MPI_SUCCESS) {
        log_error("Failed to initialize MPI");
        return false;
    }
    comm = MPI_COMM_WORLD;

    log("MPI is ready, retrieving communicator properties");
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    // old colouring code
    //    /* Now, perform the node colouring and obtain the communicator. */
    //    Log::DefaultLog.WriteInfo("Performing node colouring with colour "
    //        "%d ...", colour);
    //    // TODO: Check status?
    //    ::MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &comm);
    //    this->comm.store(comm);

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::MpiNode on %hs is %d of %d.",
        vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), rank, comm_size);

    return true;
}

bool megamol::frontend::Remote_Service::MpiNode::close() {
    auto& comm = mpi_comm.mpi_comm;

    if (comm == MPI_COMM_NULL)
        return true;

    log("Releasing MPI communicator");
    if (comm != MPI_COMM_WORLD) // can not free world
        MPI_Comm_free(&comm);
    comm = MPI_COMM_NULL;

    log("Finalising MPI");
    MPI_Finalize();

    return true;
}

bool megamol::frontend::Remote_Service::MpiNode::get_broadcast_message(megamol::remote::Message_t& message) {
    if (mpi_comm.broadcast_rank < 0) {
        log_error("(" + std::to_string(mpi_comm.rank) + ") Broadcast rank not set. Skipping.");
        return false;
    }

    const auto i_broadcast = i_do_broadcast();

    uint64_t msg_size = (i_broadcast ? message.size() : 0);
    MPI_Bcast(&msg_size, 1, MPI_UINT64_T, mpi_comm.broadcast_rank, mpi_comm.mpi_comm);

    if (!i_broadcast)
        message.resize(msg_size);

    MPI_Bcast(message.data(), msg_size, MPI_UNSIGNED_CHAR, mpi_comm.broadcast_rank, mpi_comm.mpi_comm);

    return true;
}

void megamol::frontend::Remote_Service::MpiNode::sync_barrier() {
    MPI_Barrier(mpi_comm.mpi_comm);
}

#endif // MEGAMOL_USE_MPI
