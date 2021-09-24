/*
 * MpiNode.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "MpiNode.hpp"

#include <chrono>

#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/SystemInformation.h"

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

#ifndef WITH_MPI
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

    if (isInitialised) {
        log_warning("MPI has already been initialised");
        return true;
    }
    if (comm_ != MPI_COMM_NULL) {
        log_error("MPI has not been initialised but my communicator is not MPI_COMM_NULL");
        return true;
    }
    log("Initialising MPI");

    broadcast_rank_ = broadcast_rank;
    comm_ = MPI_COMM_NULL;
    rank_ = 0;
 
    auto init_status = MPI_Init(NULL, NULL);
    if (init_status != MPI_SUCCESS) {
        log_error("Failed to initialize MPI");
        return false;
    }
    comm_ = MPI_COMM_WORLD;

    log("MPI is ready, retrieving communicator properties");
    MPI_Comm_size(comm_, &comm_size_);
    MPI_Comm_rank(comm_, &rank_);

    // old colouring code
    //    /* Now, perform the node colouring and obtain the communicator. */
    //    Log::DefaultLog.WriteInfo("Performing node colouring with colour "
    //        "%d ...", colour);
    //    // TODO: Check status?
    //    ::MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &comm);
    //    this->comm.store(comm);

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::MpiNode on %hs is %d of %d.",
        vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), rank_, comm_size_);

    return true;
}

bool megamol::frontend::Remote_Service::MpiNode::close() {
    if (comm_ == MPI_COMM_NULL)
        return true;

    log("Releasing MPI communicator");
    if (comm_ != MPI_COMM_WORLD) // can not free world
        MPI_Comm_free(&comm_);
    comm_ = MPI_COMM_NULL;

    log("Finalising MPI");
    MPI_Finalize();

    return true;
}

bool megamol::frontend::Remote_Service::MpiNode::get_broadcast_message(megamol::remote::Message_t& message) {
    if (broadcast_rank_ < 0) {
        log_error("(" + std::to_string(rank_) + ") Broadcast rank not set. Skipping.");
        return false;
    }

    const auto i_broadcast = i_do_broadcast();

    uint64_t msg_size = (i_broadcast ? message.size() : 0);
    MPI_Bcast(&msg_size, 1, MPI_UINT64_T, broadcast_rank_, comm_);

    if (!i_broadcast)
        message.resize(msg_size);

    MPI_Bcast(message.data(), msg_size, MPI_UNSIGNED_CHAR, broadcast_rank_, comm_);

    return true;
}

void megamol::frontend::Remote_Service::MpiNode::sync_barrier() {
    MPI_Barrier(comm_);
}

#endif // WITH_MPI

