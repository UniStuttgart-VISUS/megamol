/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#include "ConnectionFactory.h"

ConnectionFactory::ConnectionFactory(std::string host) : host_(std::move(host)) {
    try {
        context_ = std::make_unique<zmq::context_t>(1);
    } catch (zmq::error_t const& ex) {
        throw std::runtime_error("Error init zmq: " + std::string(ex.what()));
    }
}

ConnectionFactory::~ConnectionFactory() {
    context_->close();
    context_.reset();
}

std::unique_ptr<Connection> ConnectionFactory::createConnection(uint32_t timeout) {
    try {
        zmq::socket_t socket(*context_, zmq::socket_type::req);
        socket.set(zmq::sockopt::linger, 0);
        socket.connect(host_);
        return std::make_unique<Connection>(std::move(socket), timeout);
    } catch (zmq::error_t const& ex) {
        throw std::runtime_error("Error creating connection: " + std::string(ex.what()));
    }
}
