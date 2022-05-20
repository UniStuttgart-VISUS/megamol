#pragma once

#include <cctype>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <zmq.hpp>

class Connection {
public:
    Connection(zmq::socket_t& socket, const int timeOut);
    ~Connection();

    inline zmq::socket_t& Socket() {
        return socket;
    }
    inline zmq::socket_t const& Socket() const {
        return socket;
    }
    inline operator zmq::socket_t&() {
        return socket;
    }
    inline operator zmq::socket_t const &() const {
        return socket;
    }

    inline std::string& ActiveHost() {
        return activeHost;
    }
    inline const std::string& ActiveHost() const {
        return activeHost;
    }

    inline bool Disconnect() {
        if (!activeHost.empty()) {
            socket.disconnect(activeHost);
            activeHost.clear();
            return true;
        }
        return false;
    }
    inline bool Connected() {
        return !activeHost.empty();
    }

    std::string sendCommand(const std::string& cmd) {
        auto sent = socket.send(cmd.data(), cmd.length());
        //std::cout << "sent " << sent << "bytes";
        zmq::message_t reply;

        size_t counter = 0;
        bool have_something = false;
        while (!(have_something = socket.recv(&reply, ZMQ_DONTWAIT)) && counter < timeOut * 100) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ++counter;
        }
        if (have_something) {
            return std::string(reinterpret_cast<char*>(reply.data()), reply.size());
        } else {
            return "reply timeout, probably MegaMol was closed. Please reconnect.";
        }
    }

    inline bool Connect(const std::string& host) {
        if (!activeHost.empty())
            return false;
        //socket.setsockopt(ZMQ_SNDHWM, 0);
        socket.connect(host);
        if (!socket.connected()) {
            throw std::runtime_error("Not connected after \"connect\" returned");
        }

        activeHost = host;
        return true;
    }

private:
    zmq::socket_t& socket;
    std::string activeHost;
    int timeOut = 0;
};
