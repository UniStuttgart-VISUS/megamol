/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */
#include "Connection.h"

#include <chrono>
#include <thread>

Connection::Connection(zmq::socket_t&& socket, uint32_t timeOut) : socket_(std::move(socket)), timeOut_(timeOut) {}

std::string Connection::sendCommand(const std::string& cmd) {
    if (cmd.empty()) {
        return {};
    }
    try {
        zmq::message_t msg(cmd.cbegin(), cmd.cend());
        socket_.send(msg, zmq::send_flags::none);

        uint32_t timeCounter = 0;
        zmq::recv_result_t result;
        zmq::message_t response;

        while (!(result = socket_.recv(response, zmq::recv_flags::dontwait)).has_value() && timeCounter < timeOut_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            timeCounter += 10;
        }

        if (result.has_value()) {
            return response.to_string();
        } else {
            return "Reply timeout, probably MegaMol was closed. Please reconnect.";
        }
    } catch (zmq::error_t const& ex) {
        throw std::runtime_error("zmq send/recv error: " + std::string(ex.what()));
    }
}

Connection::~Connection() {
    socket_.close();
}
