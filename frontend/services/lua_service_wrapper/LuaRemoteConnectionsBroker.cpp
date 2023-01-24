/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "LuaRemoteConnectionsBroker.h"

#include <zmq_addon.hpp>

#include "mmcore/utility/log/Log.h"

bool megamol::frontend::LuaRemoteConnectionsBroker::Init(const std::string& broker_address, bool retry_socket_port) {
    using megamol::core::utility::log::Log;

    int retries = 0;
    int max_retries = 100 + 1;

    auto delimiter_pos = broker_address.find_last_of(':');
    std::string base_address = broker_address.substr(0, delimiter_pos + 1); // include delimiter
    std::string port_string = broker_address.substr(delimiter_pos + 1);     // ignore delimiter
    int port = std::stoi(port_string);

    // retry to start broker socket on next port until max_retries reached
    std::string address;
    do {
        port_string = std::to_string(port + retries);
        address = base_address + port_string;

        zmq::socket_t socket(zmq_context_, zmq::socket_type::router);
        socket.set(zmq::sockopt::linger, 0);
        socket.set(zmq::sockopt::rcvtimeo, 100); // message receive time out 100ms

        try {
            Log::DefaultLog.WriteInfo("LRH Server attempt socket on \"%s\"", address.c_str());
            socket.bind(address);
            Log::DefaultLog.WriteInfo("LRH Server socket opened on \"%s\"", address.c_str());

            worker_ = std::thread(&LuaRemoteConnectionsBroker::worker, this, std::move(socket));

            return true;
        } catch (std::exception& error) {
            Log::DefaultLog.WriteError("Error on LRH Server: %s", error.what());
        } catch (...) {
            Log::DefaultLog.WriteError("Error on LRH Server: unknown exception");
        }

        retries++;
    } while (retry_socket_port && retries < max_retries);

    if (retry_socket_port && retries == max_retries) {
        Log::DefaultLog.WriteInfo("LRH Server max socket port retries reached (%i). Address: %s, tried up to port: %s",
            max_retries, broker_address.c_str(), port_string.c_str());
        return false;
    }

    return true;
}

void megamol::frontend::LuaRemoteConnectionsBroker::Close() {
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
