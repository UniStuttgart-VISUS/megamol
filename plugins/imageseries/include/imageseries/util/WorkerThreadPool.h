/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace megamol::ImageSeries::util {

class WorkerThreadPool;

class Job {
public:
    using Func = std::function<void()>;

private:
    enum Status {
        WAITING = 0,
        ACTIVE,
        DONE,
        CANCELLED,
    };

    struct JobData {
        Func func;
        std::atomic_int status = ATOMIC_VAR_INIT(Status::WAITING);
        std::mutex mutex;
        std::condition_variable condition;

        bool isPending() const;
    };

public:
    Job() = default;
    Job(std::shared_ptr<JobData> jobData);

    Job(const Job&) = default;
    Job(Job&&) = default;
    Job& operator=(const Job&) = default;
    Job& operator=(Job&&) = default;

    bool await();
    bool execute();
    bool cancel();

    bool isPending() const;
    bool isInProgress() const;
    bool isDone() const;
    bool isCancelled() const;

private:
    Status getStatus() const;

    std::weak_ptr<JobData> jobData;

    friend class WorkerThreadPool;
};

class WorkerThreadPool {
public:
    WorkerThreadPool();
    ~WorkerThreadPool();

    static WorkerThreadPool& getSharedInstance();

    Job submit(Job::Func func);

    void setThreadCount(std::size_t count);
    std::size_t getThreadCount() const;

private:
    void startThreads();
    void stopThreads();

    std::shared_ptr<Job::JobData> awaitWork();
    void workerLoop();

    std::deque<std::shared_ptr<Job::JobData>> jobs;

    std::size_t threadCount = 12;
    std::vector<std::thread> threads;
    mutable std::mutex mutex;
    mutable std::mutex controlMutex;
    mutable std::condition_variable conditionQueue;
    std::atomic_bool running = ATOMIC_VAR_INIT(false);
};

} // namespace megamol::ImageSeries::util
