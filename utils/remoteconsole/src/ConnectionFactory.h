/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <zmq.hpp>

#include "Connection.h"

class ConnectionFactory {
public:
    explicit ConnectionFactory(std::string host);
    ~ConnectionFactory();

    ConnectionFactory(ConnectionFactory const& other) = delete;
    ConnectionFactory(ConnectionFactory&& other) = delete;
    ConnectionFactory& operator=(ConnectionFactory const& other) = delete;
    ConnectionFactory& operator=(ConnectionFactory&& other) = delete;

    std::unique_ptr<Connection> createConnection(uint32_t timeout);

private:
    std::string host_;
    std::unique_ptr<zmq::context_t> context_;
};
