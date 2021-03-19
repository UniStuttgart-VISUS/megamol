/*
 * RenderNode.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "Remote_Service.hpp"
#include "comm/DistributedProto.h"
#include "comm/FBOCommFabric.h"

#include "ThreadWorker.h"

#include <mutex>
#include <atomic>
#include <condition_variable>

struct megamol::frontend::Remote_Service::RenderNode {
    RenderNode() = default;
    //, bool use_mpi, bool sync_data_sources_mpi, int broadcast_rank_mpi);
    ~RenderNode();

    bool start_receiver(int listen_port = 62562);
    bool close_receiver();
    bool await_message(megamol::remote::Message_t& result, unsigned int timeout_ms = 1000);

private:
    megamol::remote::FBOCommFabric receiver_comm_{std::make_unique<megamol::remote::ZMQCommFabric>(zmq::socket_type::pull)};
    megamol::frontend_resources::ThreadWorker receiver_thread_;

    void receiver_thread_loop();

    megamol::remote::Message_t recv_msgs_;
    mutable std::mutex recv_msgs_mtx_;
    std::condition_variable data_received_cond_;
    std::atomic<bool> data_has_changed_ = false;
};

