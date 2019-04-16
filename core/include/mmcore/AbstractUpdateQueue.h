#pragma once

#include <mutex>
#include <queue>

namespace megamol {
namespace core {

template <typename T> class AbstractUpdateQueue {
public:
    std::unique_lock<std::mutex> AcquireLock() { return std::unique_lock<std::mutex>(update_lock_); }

    std::unique_lock<std::mutex> AcquireDeferredLock() {
        return std::unique_lock<std::mutex>(update_lock_, std::defer_lock);
    }

    void Push(T&& el) { update_queue_.push(el); }

    void Push(T const& el) { update_queue_.push(el); }

    typename std::queue<T>::value_type Get() {
        T el = update_queue_.front();
        update_queue_.pop();
        return el;
    }

    template <typename... Args> void Emplace(Args&&... vals) { update_queue_.emplace(std::forward<Args>(vals)...); }

    bool Empty() { return update_queue_.empty(); }

    typename std::queue<T>::size_type Size() { return update_queue_.size(); }

private:
    std::mutex update_lock_;

    std::queue<T> update_queue_;
}; // end class AbstractUpdateQueue

} // end namespace core
} // end namespace megamol