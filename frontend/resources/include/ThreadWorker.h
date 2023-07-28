/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>

namespace megamol::frontend_resources {

struct ThreadSignaler {
    std::atomic<bool> running = false;

    void start() {
        running.store(true, std::memory_order_release);
    }
    void stop() {
        running.store(false, std::memory_order_release);
    }
    bool is_running() {
        return running.load(std::memory_order_acquire);
    }
};

struct ThreadWorker {
    ThreadSignaler signal;
    std::thread thread;

    void join() {
        signal.stop();
        if (thread.joinable())
            thread.join();
    }
};


template<typename Item>
class threadsafe_queue {
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

    bool empty() const {
        guard;
        return m_queue.empty();
    }

#undef guard
private:
    std::queue<Item> m_queue;
    mutable std::mutex m_mutex;
};


} // namespace megamol::frontend_resources
