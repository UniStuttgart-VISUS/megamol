/*
 * HeadNode.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "HeadNode.hpp"

//#include <chrono>

#include "mmcore/utility/log/Log.h"

using namespace megamol::remote;

bool megamol::frontend::Remote_Service::HeadNode::send(megamol::remote::Message_t const& data) {
    if (!comm_thread_.signal.is_running() || data.empty())
        return false;

    // the Remote_Service fills the message data according to the used convention
    // we are only responsible to send the data here

    std::lock_guard<std::mutex> guard(send_buffer_guard_);
    send_buffer_.clear();
    send_buffer_.reserve(data.size());
    send_buffer_.insert(send_buffer_.end(), data.begin(), data.end());
    send_buffer_has_changed_.store(true);

    return true;
}

bool megamol::frontend::Remote_Service::HeadNode::start_server(std::string const& send_to_address) {
    try {
        close_server();
        this->comm_fabric_ = FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::push));
        this->comm_fabric_.Connect(send_to_address);
        this->comm_thread_.thread = std::thread{[&]() { this->comm_thread_loop(); }};
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::HeadNode: Communication thread started.");
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Remote_Service::HeadNode: Could not initialize communication thread.");
        return false;
    }

    return true;
}

bool megamol::frontend::Remote_Service::HeadNode::close_server() {
    if (comm_thread_.signal.is_running()) {
        comm_fabric_.Disconnect();
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::HeadNode: Joining sender thread.");
    comm_thread_.signal.stop();
    comm_thread_.join();

    return true;
}

void megamol::frontend::Remote_Service::HeadNode::comm_thread_loop() {
    comm_thread_.signal.start();

    try {
        while (comm_thread_.signal.is_running()) {
            // TODO: promise, future? events queue?
            if (send_buffer_has_changed_.load()) {
                std::lock_guard<std::mutex> lock(send_buffer_guard_);
                comm_fabric_.Send(send_buffer_, send_type::SEND);
                send_buffer_.clear();
                send_buffer_has_changed_.store(false);
            } /*else {
                comm_fabric_.Send(null_buf, send_type::SEND);
            }*/

            // using namespace std::chrono_literals;
            // std::this_thread::sleep_for(1000ms / 120);
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Remote_Service::HeadNode: Error during communication;");
    }
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Remote_Service::HeadNode: Exiting sender loop.");

    comm_thread_.signal.stop();
}

megamol::frontend::Remote_Service::HeadNode::~HeadNode() {
    close_server();
}

