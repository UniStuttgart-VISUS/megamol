/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "Remote_Service.hpp"
#include "ThreadWorker.h"
#include "comm/DistributedProto.h"
#include "comm/FBOCommFabric.h"

struct megamol::frontend::Remote_Service::RenderNode {
    RenderNode() = default;
    //, bool use_mpi, bool sync_data_sources_mpi, int broadcast_rank_mpi);
    ~RenderNode();

    bool start_receiver(std::string const& receive_from_address);
    bool close_receiver();
    bool await_message(megamol::remote::Message_t& result, unsigned int timeout_ms = 1000);

private:
    megamol::remote::FBOCommFabric receiver_comm_{
        std::make_unique<megamol::remote::ZMQCommFabric>(zmq::socket_type::pull)};
    megamol::frontend_resources::ThreadWorker receiver_thread_;

    void receiver_thread_loop();

    megamol::remote::Message_t recv_msgs_;
    mutable std::mutex recv_msgs_mtx_;
    std::condition_variable data_received_cond_;
    std::atomic<bool> data_has_changed_ = false;
};
