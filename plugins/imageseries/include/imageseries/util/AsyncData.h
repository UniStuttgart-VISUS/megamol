/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "WorkerThreadPool.h"

#include <atomic>
#include <functional>
#include <memory>

namespace megamol::ImageSeries::util {

template<typename Data>
class AsyncData {
public:
    using DataPtr = std::shared_ptr<Data>;
    using DataProvider = std::function<DataPtr()>;
    using Hash = std::uint32_t;

    AsyncData(DataProvider provider, std::size_t byteSize) : byteSize(byteSize), hash(computeHash()) {
        job = WorkerThreadPool::getSharedInstance().submit([this, provider]() { data = provider(); });
    }

    AsyncData(DataPtr initData) : AsyncData(initData, initData != nullptr ? computeByteSize(initData) : 0) {}

    AsyncData(DataPtr initData, std::size_t initByteSize)
            : byteSize(initByteSize)
            , data(initData)
            , hash(initData != nullptr ? computeHash() : 0) {}

    ~AsyncData() {
        // Try to cancel job
        if (!job.cancel()) {
            // If not possible, wait for its completion
            job.await();
        }
    }

    bool isWaiting() const {
        return job.isPending();
    }

    bool isFinished() const {
        return !job.isPending();
    }

    bool isValid() const {
        return isFinished() && data;
    }

    bool isFailed() const {
        return isFinished() && !data;
    }

    std::size_t getByteSize() const {
        return byteSize;
    }

    Hash getHash() const {
        return hash;
    }

    DataPtr tryGetData() const {
        return isFinished() ? data : nullptr;
    }

    DataPtr getData() const {
        job.execute();
        return data;
    }

private:
    Hash computeHash() {
        // TODO use an actual hash function instead of a counter!
        static std::atomic<Hash> currentHash = ATOMIC_VAR_INIT(1);
        return currentHash++;
    }

    std::size_t byteSize = 0;
    DataPtr data;
    Hash hash = 0;

    mutable util::Job job;
};


} // namespace megamol::ImageSeries::util
