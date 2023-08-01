/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <mutex>

namespace megamol::core {

class lockable {
public:
    void lock() {
        lock_.lock();
    }

    bool try_lock() {
        return lock_.try_lock();
    }

    void unlock() {
        lock_.unlock();
    }

    std::mutex::native_handle_type native_handle() {
        return lock_.native_handle();
    }

    virtual ~lockable() = default;

private:
    mutable std::mutex lock_;
}; // end class lockable

} // namespace megamol::core
