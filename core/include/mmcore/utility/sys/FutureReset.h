/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>

namespace megamol {
namespace core {
namespace utility {
namespace sys {

template<typename T>
class FutureReset {
public:
    FutureReset() : promise_{std::make_unique<std::promise<T>>()}, future_{promise_->get_future()} {}

    /*FutureReset(FutureReset const& rhs) = delete;

    FutureReset& operator=(FutureReset const& rhs) = delete;*/

    // FutureReset(FutureReset&& rhs) noexcept : FutureReset{} {
    //    swap(promise_, rhs.promise_);
    //    swap(future_, rhs.future_);
    //    //swap(exchange_lock_, rhs.exchange_lock_);
    //}

    // FutureReset& operator=(FutureReset&& rhs) noexcept {
    //    swap(promise_, rhs.promise_);
    //    swap(future_, rhs.future_);
    //    //swap(exchange_lock_, rhs.exchange_lock_);
    //    return *this;
    //}

    FutureReset* GetPtr() {
        return this;
    }

    FutureReset const* GetPtr() const {
        return this;
    }

    void SetPromise(T const& value) {
        std::unique_lock<std::mutex> exchange_guard(exchange_lock_);
        promise_->set_value(value);
    }

    void SetPromise(T&& value) {
        std::unique_lock<std::mutex> exchange_guard(exchange_lock_);
        promise_->set_value(std::forward<T>(value));
    }

    T GetAndReset() {
        std::unique_lock<std::mutex> lg(exchange_lock_);
        T ret = future_.get();
        reset();
        return ret;
    }

    template<typename REP, typename PERIOD>
    std::future_status WaitFor(std::chrono::duration<REP, PERIOD> const& timeout_duration) const {
        return future_.wait_for(timeout_duration);
    }

private:
    void reset() {
        promise_.reset(new std::promise<T>());
        future_ = promise_->get_future();
    }

    std::unique_ptr<std::promise<T>> promise_;

    std::future<T> future_;

    std::mutex exchange_lock_;

    std::condition_variable cv;
}; // end class FutureReset

} // end namespace sys
} // end namespace utility
} // end namespace core
} // end namespace megamol
