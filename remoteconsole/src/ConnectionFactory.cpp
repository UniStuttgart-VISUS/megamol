/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#include "ConnectionFactory.h"

ConnectionFactory::ConnectionFactory(std::string host) : host_(std::move(host)) {
    try {
        context_ = std::make_unique<zmq::context_t>(1);
        pre_socket_ = std::make_unique<zmq::socket_t>(*context_, ZMQ_REQ);
        pre_socket_->setsockopt(ZMQ_LINGER, 0);
        pre_socket_->connect(host_);
    } catch (zmq::error_t const& ex) { throw std::runtime_error("Error init zmq: " + std::string(ex.what())); }
}

ConnectionFactory::~ConnectionFactory() {
    pre_socket_->close();
    pre_socket_.reset();
    context_->close();
    context_.reset();
}

std::unique_ptr<Connection> ConnectionFactory::createConnection(uint32_t timeout) {
    try {
        const std::string hello = "ola";
        zmq::message_t hello_msg(hello.cbegin(), hello.cend());
        pre_socket_->send(hello_msg, zmq::send_flags::none);

        zmq::message_t response;
        auto result = pre_socket_->recv(response, zmq::recv_flags::none);
        if (!result.has_value()) {
            throw std::runtime_error("Error receiving remote port number!");
        }

        int port = std::stoi(response.to_string());
        port = std::clamp(port, 0, 65535);
        const auto portPos = host_.find_last_of(':');
        std::string newHost = host_.substr(0, portPos) + ":" + std::to_string(port);

        zmq::socket_t socket(*context_, ZMQ_PAIR);
        socket.setsockopt(ZMQ_LINGER, 0);
        socket.connect(newHost);
        return std::make_unique<Connection>(std::move(socket), timeout);
    } catch (zmq::error_t const& ex) {
        throw std::runtime_error("Error creating connection: " + std::string(ex.what()));
    }
}
