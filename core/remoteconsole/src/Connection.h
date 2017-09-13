#pragma once
#include <zmq.hpp>
#include <string>
#include <cctype>
#include <vector>

class Connection {
public:
    Connection(zmq::socket_t& socket);
    ~Connection();

    inline zmq::socket_t& Socket() { return socket; }
    inline zmq::socket_t const & Socket() const { return socket; }
    inline operator zmq::socket_t&() { return socket; }
    inline operator zmq::socket_t const &() const { return socket; }

    inline std::string& ActiveHost() { return activeHost; }
    inline const std::string& ActiveHost() const { return activeHost; }

    inline bool Disconnect() {
        if (!activeHost.empty()) {
            socket.disconnect(activeHost);
            activeHost.clear();
            protocols.clear();
            return true;
        }
        return false;
    }
    inline bool Connected() {
        return !activeHost.empty();
    }

    std::string sendCommand(const std::string& cmd) {
        socket.send(cmd.data(), cmd.length());
        zmq::message_t reply;
        socket.recv(&reply);
        return std::string(reinterpret_cast<char*>(reply.data()), reply.size());
    }

    inline bool Connect(const std::string &host) {
        if (!activeHost.empty()) return false;

        socket.connect(host);
        if (!socket.connected()) {
            throw std::runtime_error("Not connected after \"connect\" returned");
        }

        bool supportMMSPR1 = false;

        std::string protocols = sendCommand("PROTOCOLS");
        size_t i = 0;
        size_t l = protocols.length();
        while (i < l) {
            // skip leading whitespace
            while ((i < l) && (std::isspace(protocols[i]))) ++i;
            if (i == l) break;
            size_t wordStart = i;
            while ((i < l) && (!std::isspace(protocols[i]))) ++i;
            if (i > wordStart) {
                std::string protocol(protocols.data() + wordStart, protocols.data() + i);
                this->protocols.push_back(protocol);
            }
        }

        if (!supportProtocol("MMSPR1")) {
            socket.disconnect(host);
            throw std::runtime_error("Connection does not support required protocol MMSPR1");
        }

        activeHost = host;
        return true;
    }

    /**
     * Answer whether or not the requested protocol is supported by the connected host
     *
     * @param proto The protocol identifier string
     *
     * @return True if the requested protocol is supported
     */
    inline bool supportProtocol(const char* proto) const {
        return std::find(protocols.begin(), protocols.end(), proto) != protocols.end();
    }

private:
    zmq::socket_t& socket;
    std::string activeHost;
    // protocols supported by the connected host
    std::vector<std::string> protocols;
};

