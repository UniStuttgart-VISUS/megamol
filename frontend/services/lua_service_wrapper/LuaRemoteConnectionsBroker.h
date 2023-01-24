/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <future>
#include <list>
#include <string>
#include <thread>

#include <zmq.hpp>

#include "ThreadWorker.h"

namespace megamol::frontend {

class LuaRemoteConnectionsBroker {
public:
    struct LuaRequest {
        std::string request;
        std::promise<std::string> answer_promise;
    };

    bool RequestQueueEmpty() const {
        return request_queue_.empty();
    }

    std::queue<LuaRequest> GetRequestQueue() {
        return request_queue_.pop_queue();
    }

    bool Init(std::string const& broker_address, bool retry_socket_port = false);

    void Close();

private:
    void worker(zmq::socket_t&& socket);

    struct LuaResponse {
        std::string client_id;
        std::future<std::string> future;
    };

    zmq::context_t zmq_context_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    frontend_resources::threadsafe_queue<LuaRequest> request_queue_;
    std::list<LuaResponse> response_queue_;
};

} // namespace megamol::frontend
