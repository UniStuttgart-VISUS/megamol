/*
 * LuaRemoteConnectionsBroker.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ThreadWorker.h"
#include <future>

#include "mmcore/utility/ZMQContextUser.h"

namespace megamol {
namespace frontend_resources {

class LuaRemoteConnectionsBroker {
public:
    ThreadWorker broker_worker;
    std::list<ThreadWorker> lua_workers;

    void close() {
        broker_worker.join();
        for (auto& w : lua_workers) {
            w.join();
        }
    }

    struct LuaRequest {
        std::string request;
        std::reference_wrapper<std::promise<std::string>> answer_promise;
    };
    threadsafe_queue<LuaRequest> request_queue;

    std::queue<LuaRequest> get_request_queue() { return std::move(request_queue.pop_queue()); }

    zmq::context_t zmq_context;

    // connection broker starts threads that execute incoming lua commands
    bool spawn_connection_broker(std::string broker_address, bool retry_socket_port = false) {
        assert(broker_worker.signal.is_running() == false);

        int retries = 0;
        int max_retries = 100+1;

        auto delimiter_pos = broker_address.find_last_of(':');
        std::string base_address = broker_address.substr(0, delimiter_pos+1); // include delimiter
        std::string port_string = broker_address.substr(delimiter_pos+1); // ignore delimiter
        int port = std::stoi(port_string);

        // retry to start broker socket on next port until max_retries reached
        std::string address;
        do {
            port_string = std::to_string(port + retries);
            address = base_address + port_string;

            std::promise<bool> socket_feedback;
            auto socket_ok = socket_feedback.get_future();

            auto thread =
                std::thread([&]() { this->connection_broker_routine(zmq_context, address, broker_worker.signal, socket_feedback); });

            // wait for lua socket to start successfully or fail - so we can propagate the fail and stop megamol execution
            socket_ok.wait();

            if (!socket_ok.get() || !broker_worker.signal.is_running()) {
                if(retry_socket_port) {
                    // do loop one more time
                    retries++;

                    if(thread.joinable())
                        thread.join();
                }
                else {
                    // no retries, just fail
                    return false;
                }
            }
            else {
                // broker thread started successfully, so break out of retry loop
                broker_worker.thread = std::move(thread);
                break;
            }
        } while(retries < max_retries);

        if (retry_socket_port && retries == max_retries) {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(("LRH Server max socket port retries reached (" + std::to_string(max_retries) + "). Address: " + broker_address + ", tried up to port: " + port_string).c_str());
            return false;
        }

        return true;
    }

    std::string spawn_lua_worker(std::string const& request) {
        if (request.empty()) return std::string("Null Command.");

        std::promise<int> port_promise;
        auto port_future = port_promise.get_future();

        lua_workers.emplace_back();
        auto& worker = lua_workers.back();
        worker.thread = std::thread([&]() { this->lua_console_routine(zmq_context, worker.signal, port_promise); });

        while (broker_worker.signal.is_running() &&
               port_future.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        int port = port_future.get();

        megamol::core::utility::log::Log::DefaultLog.WriteInfo("LRH: generated PAIR socket on port %i", port);
        return std::to_string(port);
    }

    void connection_broker_routine(zmq::context_t& zmq_context, std::string const& address, ThreadSignaler& signal, std::promise<bool>& socket_ok) {
        using megamol::core::utility::log::Log;

        zmq::socket_t socket(zmq_context, ZMQ_REP);
        Log::DefaultLog.WriteInfo("LRH Server attempt socket on \"%s\"", address.c_str());

        try {
            socket.bind(address);
            socket.setsockopt(ZMQ_RCVTIMEO, 100); // message receive time out 100ms

            Log::DefaultLog.WriteInfo("LRH Server socket opened on \"%s\"", address.c_str());

            signal.start();
            socket_ok.set_value(true);
            while (signal.is_running()) {
                zmq::message_t request;

                if (socket.recv(&request, ZMQ_DONTWAIT)) {
                    std::string request_str(reinterpret_cast<char*>(request.data()), request.size());
                    std::string reply = spawn_lua_worker(request_str);
                    socket.send(reply.data(), reply.size());
                } else {
                    // no messages available ATM
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }

        } catch (std::exception& error) {
            Log::DefaultLog.WriteError("Error on LRH Server: %s", error.what());
            socket_ok.set_value(false);
        } catch (...) {
            Log::DefaultLog.WriteError("Error on LRH Server: unknown exception");
            socket_ok.set_value(false);
        }

        try {
            socket.close();
        } catch (...) {
            Log::DefaultLog.WriteInfo("LRH Server socket close threw exception");
        }
        Log::DefaultLog.WriteInfo("LRH Server socket closed");
    }

    void lua_console_routine(
        zmq::context_t& zmq_context, ThreadSignaler& signal, std::promise<int>& socket_port_feedback) {
        using megamol::core::utility::log::Log;

        auto socket = zmq::socket_t(zmq_context, zmq::socket_type::pair);
        socket.bind("tcp://*:0");
        std::array<char, 1024> opts;
        size_t len = opts.size();
        socket.getsockopt(ZMQ_LAST_ENDPOINT, opts.data(), &len);
        std::string endp(opts.data());
        const auto portPos = endp.find_last_of(":");
        const auto portStr = endp.substr(portPos + 1, -1);
        socket_port_feedback.set_value(std::atoi(portStr.c_str()));

        try {
            signal.start();
            while (signal.is_running()) {
                zmq::message_t request;

                if (!socket.connected()) break;

                if (socket.recv(&request, ZMQ_DONTWAIT)) {
                    std::string request_str(reinterpret_cast<char*>(request.data()), request.size());
                    std::promise<std::string> promise;
                    std::future<std::string> future = promise.get_future();

                    request_queue.push(LuaRequest{request_str, std::ref(promise)});
                    std::string reply{"no response from LuaAPI at main thread"};

                    // wait for response from main thread
                    while (signal.is_running()) {
                        if (future.wait_for(std::chrono::milliseconds(32)) == std::future_status::ready) {
                            reply = future.get();
                            break;
                        }
                    }

                    const auto num_sent = socket.send(reply.data(), reply.size());
#ifdef LRH_ANNOYING_DETAILS
                    if (num_sent == reply.size()) {
                        megamol::core::utility::log::Log::DefaultLog.WriteInfo("LRH: sending looks OK");
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("LRH: send failed");
                    }
#endif
                } else {
                    // no messages available ATM
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            signal.stop();

        } catch (std::exception& error) {
            Log::DefaultLog.WriteError("Error on LRH Pair Server: %s", error.what());

        } catch (...) {
            Log::DefaultLog.WriteError("Error on LRH Pair Server: unknown exception");
        }

        try {
            socket.close();
        } catch (...) {
        }
        Log::DefaultLog.WriteInfo("LRH Pair Server socket closed");
    }
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
