#pragma once
#include <mutex>

namespace megamol {
namespace core {

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

} // end namespace core
} // end namespace megamol
