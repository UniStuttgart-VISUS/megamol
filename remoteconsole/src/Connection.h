/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <string>

#include <zmq.hpp>

class Connection {
public:
    Connection(zmq::socket_t&& socket, uint32_t timeOut);
    ~Connection();

    Connection(Connection const& other) = delete;
    Connection(Connection&& other) = delete;
    Connection& operator=(Connection const& other) = delete;
    Connection& operator=(Connection&& other) = delete;

    std::string sendCommand(std::string const& cmd);

private:
    zmq::socket_t socket_;
    uint32_t timeOut_ = 0;
};
