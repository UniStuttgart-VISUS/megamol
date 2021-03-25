/*
 * RenderNode.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "RenderNode.hpp"

#include <chrono>

#include "mmcore/utility/log/Log.h"

using namespace megamol::remote;

megamol::frontend::Remote_Service::RenderNode::~RenderNode() {
    close_receiver();
}

bool megamol::frontend::Remote_Service::RenderNode::start_receiver(std::string const& receive_from_address) {
    close_receiver();

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::RenderNode: Starting listener on %s.", receive_from_address.c_str());

    try {
        this->receiver_comm_ = FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::pull));
        this->receiver_comm_.Bind(receive_from_address);
        receiver_thread_.thread = std::thread{[&]() { receiver_thread_loop(); }};
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::RenderNode: Receiver thread started.");
    }
    catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Remote_Service::RenderNode: Could not initialize receiver thread.");

    }
    return true;
}

bool megamol::frontend::Remote_Service::RenderNode::close_receiver() {
    if (receiver_thread_.signal.is_running()) {
        receiver_comm_.Disconnect();
    }
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::RenderNode: Joining receiver thread.");
    receiver_thread_.signal.stop();
    receiver_thread_.join();

    return true;
}

void megamol::frontend::Remote_Service::RenderNode::receiver_thread_loop() {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::RenderNode: Starting receiver loop.");

    try {
        receiver_thread_.signal.start();
        while (receiver_thread_.signal.is_running()) {
            Message_t buf = {'r', 'e', 'q'};

            while (!receiver_comm_.Recv(buf, recv_type::RECV) && receiver_thread_.signal.is_running()) {
                //megamol::core::utility::log::Log::DefaultLog.WriteWarn("RendernodeView: Failed to recv message.");
            }

            if (!receiver_thread_.signal.is_running())
                break;

            {
                std::unique_lock<std::mutex> lock(recv_msgs_mtx_);
                recv_msgs_.insert(recv_msgs_.end(), buf.begin(), buf.end());
                data_has_changed_.store(true);
            }
            data_received_cond_.notify_all();
            // using namespace std::chrono_literals;
            // std::this_thread::sleep_for(1000ms / 60);
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Remote_Service::RenderNode: Error during communication.");
    }    
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::RenderNode: Exiting receiver loop.");
}


bool megamol::frontend::Remote_Service::RenderNode::await_message(megamol::remote::Message_t& result, unsigned int timeout_ms) {
    using namespace std::chrono_literals;

    std::unique_lock<std::mutex> lock(recv_msgs_mtx_);
    if (data_received_cond_.wait_for(lock, timeout_ms * 1ms, [&]() -> bool { return data_has_changed_.load(); })) {
        if (!data_has_changed_.load())
            return false;

        result.resize(recv_msgs_.size());
        std::copy(recv_msgs_.begin(), recv_msgs_.end(), result.begin());
        data_has_changed_.store(false);
        recv_msgs_.clear();
        return true;
    } else {
        return false;
    }
}

