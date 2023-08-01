/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <mutex>

#include "Remote_Service.hpp"
#include "ThreadWorker.h"
#include "comm/DistributedProto.h"
#include "comm/FBOCommFabric.h"

struct megamol::frontend::Remote_Service::HeadNode {
    HeadNode() = default;
    ~HeadNode();

    // "Sends custom lua command to the RendernodeView"
    bool send(megamol::remote::Message_t const& data);

    // "Start listening to port."
    // "Address of headnode in ZMQ syntax (e.g. \"tcp://127.0.0.1:33333\")"
    bool start_server(std::string const& send_to_address);
    bool close_server();

private:
    // before sending data, it is collected in this buffer
    // at some point comm_fabric_ sends the contents of this buffer
    megamol::remote::Message_t send_buffer_;
    mutable std::mutex send_buffer_guard_;
    std::atomic<bool> send_buffer_has_changed_ = false;

    // FBOCommFabric encapsulates either MPI or ZMQ communication
    // though called 'FBO' there is not much 'FBO' in the comm
    megamol::remote::FBOCommFabric comm_fabric_{
        std::make_unique<megamol::remote::ZMQCommFabric>(zmq::socket_type::push)};

    megamol::frontend_resources::ThreadWorker comm_thread_;
    void comm_thread_loop();
};
