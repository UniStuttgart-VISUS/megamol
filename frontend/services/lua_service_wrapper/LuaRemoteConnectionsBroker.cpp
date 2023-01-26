/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "LuaRemoteConnectionsBroker.h"

#include <zmq_addon.hpp>

#include "mmcore/utility/log/Log.h"

megamol::frontend::LuaRemoteConnectionsBroker::LuaRemoteConnectionsBroker(
    const std::string& broker_address, int max_retries) {
    using megamol::core::utility::log::Log;

    const auto delimiter_pos = broker_address.find_last_of(':');
    const std::string base_address = broker_address.substr(0, delimiter_pos + 1);
    const int base_port = std::stoi(broker_address.substr(delimiter_pos + 1));

    zmq::socket_t socket(zmq_context_, zmq::socket_type::router);
    socket.set(zmq::sockopt::linger, 0);
    socket.set(zmq::sockopt::rcvtimeo, 100); // message receive time out 100ms

    // retry to start broker socket on next port until max_retries reached
    int retry = 0;
    do {
        const std::string address = base_address + std::to_string(base_port + retry);

        try {
            socket.bind(address);
            Log::DefaultLog.WriteInfo("LRH Server socket opened on \"%s\"", address.c_str());
            worker_ = std::thread(&LuaRemoteConnectionsBroker::worker, this, std::move(socket));
            return;
        } catch (zmq::error_t& ex) {
            if (ex.num() != EADDRINUSE) {
                throw;
            }
        }
        retry++;
    } while (retry <= max_retries);

    throw std::runtime_error("LRH Server could not bind address \"y" + broker_address +
                             "\", tried up to port: " + std::to_string(base_port + retry - 1));
}

megamol::frontend::LuaRemoteConnectionsBroker::~LuaRemoteConnectionsBroker() {
    stop_ = true;
    worker_.join();
}

void megamol::frontend::LuaRemoteConnectionsBroker::worker(zmq::socket_t&& socket) {
    using megamol::core::utility::log::Log;

    while (!stop_) {
        // Receive
        zmq::multipart_t request_msg;
        while (request_msg.recv(socket, ZMQ_DONTWAIT)) {
            LuaResponse response;

            // With router socket there should be always 3 messages, but do extra safety check.
            response.client_id = (request_msg.size() >= 1) ? request_msg[0].to_string() : "";
            std::string request_str = (request_msg.size() >= 3) ? request_msg[2].to_string() : "";

            std::promise<std::string> promise;
            response.future = promise.get_future();

            response_queue_.emplace_back(std::move(response));
            request_queue_.push(LuaRequest{request_str, std::move(promise)});
        }

        // Send
        auto it = response_queue_.begin();
        while (it != response_queue_.end()) {
            if (it->future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                zmq::multipart_t response_msg;
                response_msg.addstr(it->client_id);    // Client
                response_msg.addmem(nullptr, 0);       // Separator
                response_msg.addstr(it->future.get()); // Message for REQ socket

                response_msg.send(socket);

                it = response_queue_.erase(it);
            } else {
                it++;
            }
        }

        // Reduce CPU load
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    try {
        socket.close();
    } catch (...) {
        Log::DefaultLog.WriteInfo("LRH Server socket close threw exception");
    }
    Log::DefaultLog.WriteInfo("LRH Server socket closed");
}
