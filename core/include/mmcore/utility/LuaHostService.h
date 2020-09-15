#pragma once

#include <string>
#include <thread>
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/AbstractService.h"
//#include "CommandFunctionPtr.h"
#include <atomic>
#include <map>
#include "mmcore/utility/ZMQContextUser.h"

namespace megamol {
namespace core {
namespace utility {

class LuaHostService : public core::AbstractService {
public:
    static unsigned int ID;

    virtual const char* Name() const { return "LuaRemote"; }

    LuaHostService(core::CoreInstance& core);
    virtual ~LuaHostService();

    virtual bool Initalize(bool& autoEnable);
    virtual bool Deinitialize();

    inline const std::string& GetAddress(void) const { return address; }
    void SetAddress(const std::string& ad);

protected:
    virtual bool enableImpl();
    virtual bool disableImpl();

private:
    void serve();
    void servePair();
    std::string makeAnswer(const std::string& req);
    std::string makePairAnswer(const std::string& req) const;
    std::atomic<int> lastPairPort;

    // ModuleGraphAccess mgAccess;
    ZMQContextUser::ptr context;

    std::thread serverThread;
    std::vector<std::thread> pairThreads;
    bool serverRunning;

    std::string address;
};


} /* namespace utility */
} /* namespace core */
} /* namespace megamol */

#include <atomic>
#include <future>
#include <queue>
#include <thread>
namespace megamol::core::utility {
template <typename Item> class threadsafe_queue {
#define guard std::lock_guard<std::mutex> lock(m_mutex);

public:
    void push(Item&& i) {
        guard;
        m_queue.emplace(std::move(i));
    }

    std::queue<Item> pop_queue() {
        guard;
        std::queue<Item> q = std::move(m_queue);
        m_queue = {};
        return std::move(q);
    }

    bool empty() {
        guard;
        return m_queue.empty();
    }

#undef guard
private:
    std::queue<Item> m_queue;
    mutable std::mutex m_mutex;
};

class LuaHostNetworkConnectionsBroker {
public:
    struct ThreadSignaler {
        std::atomic<bool> running = false;

        void start() { running.store(true, std::memory_order_release); }
        void stop() { running.store(false, std::memory_order_release); }
        bool is_running() { return running.load(std::memory_order_acquire); }
    };

    std::string broker_address;
    struct Worker {
        ThreadSignaler signal;
        std::thread thread;

        void join() {
            signal.stop();
            if (thread.joinable())
                thread.join();
        }
    };
    Worker broker_worker;
    std::list<Worker> lua_workers;

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
    bool spawn_connection_broker() {
        assert(broker_worker.signal.is_running() == false);

        std::promise<bool> socket_feedback;
        auto socket_ok = socket_feedback.get_future();

        broker_worker.thread =
            std::thread([&]() { this->connection_broker_routine(zmq_context, broker_address, broker_worker.signal, socket_feedback); });

        // wait for lua socket to start successfully or fail - so we can propagate the fail and stop megamol execution
        socket_ok.wait();

       if (!socket_ok.get() || !broker_worker.signal.is_running()) {
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

} // namespace megamol::core::utility
