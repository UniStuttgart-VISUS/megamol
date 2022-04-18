#include "imageseries/util/WorkerThreadPool.h"

namespace megamol::ImageSeries::util {

bool Job::JobData::isPending() const {
    int stat = this->status;
    return stat == Status::WAITING || stat == Status::ACTIVE;
}

Job::Job(std::shared_ptr<JobData> jobData) : jobData(jobData) {}

bool Job::await() {
    if (auto data = jobData.lock()) {
        if (data->isPending()) {
            std::unique_lock<std::mutex> lock(data->mutex);
            data->condition.wait(lock, [&] { return data->isPending(); });
        }
        return data->status == Status::DONE;
    } else {
        return false;
    }
}

bool Job::execute() {
    if (auto data = jobData.lock()) {
        if (!data->isPending()) {
            return data->status == Status::DONE;
        }

        std::unique_lock<std::mutex> lock(data->mutex);
        switch (data->status) {
        case Status::WAITING:
            // Execute immediately on current thread
            data->status = Status::ACTIVE;
            data->func();
            data->status = Status::DONE;
            return true;

        case Status::ACTIVE:
            data->condition.wait(lock, [&] { return data->isPending(); });
            return data->status == Status::DONE;

        case Status::DONE:
            return true;

        case Status::CANCELLED:
        default:
            return false;
        }
    } else {
        return false;
    }
}

bool Job::cancel() {
    if (auto data = jobData.lock()) {
        if (!data->isPending()) {
            // Job already cancelled
            return true;
        }

        std::unique_lock<std::mutex> lock(data->mutex);
        int pendingStatus = Status::WAITING;
        return data->status.compare_exchange_strong(pendingStatus, Status::CANCELLED);
    } else {
        // Job no longer exists -> cancelled
        return true;
    }
}

bool Job::isPending() const {
    auto status = getStatus();
    return status == Status::WAITING || status == Status::ACTIVE;
}

bool Job::isInProgress() const {
    return getStatus() == Status::ACTIVE;
}

bool Job::isDone() const {
    return getStatus() == Status::DONE;
}

bool Job::isCancelled() const {
    return getStatus() == Status::CANCELLED;
}

Job::Status Job::getStatus() const {
    if (auto data = jobData.lock()) {
        return static_cast<Job::Status>(static_cast<int>(data->status));
    } else {
        return Status::CANCELLED;
    }
}


WorkerThreadPool::WorkerThreadPool() {
    startThreads();
}

WorkerThreadPool::~WorkerThreadPool() {
    std::unique_lock<std::mutex> lock(controlMutex);

    // Remove all threads
    stopThreads();

    // Cancel all pending jobs
    for (auto& job : jobs) {
        if (job->status != Job::Status::CANCELLED) {
            {
                std::unique_lock<std::mutex> lock(job->mutex);
                job->status = Job::Status::CANCELLED;
            }
            job->condition.notify_all();
        }
    }
}

WorkerThreadPool& WorkerThreadPool::getSharedInstance() {
    static WorkerThreadPool pool;
    return pool;
}

Job WorkerThreadPool::submit(Job::Func func) {
    auto jobData = std::make_shared<Job::JobData>();
    jobData->func = std::move(func);

    {
        std::unique_lock<std::mutex> lock(mutex);
        jobs.push_back(jobData);
    }

    conditionQueue.notify_one();
    return Job(jobData);
}

void WorkerThreadPool::setThreadCount(std::size_t count) {
    std::unique_lock<std::mutex> lock(controlMutex);
    if (threadCount != count) {
        stopThreads();
        threadCount = count;
        startThreads();
    }
}

std::size_t WorkerThreadPool::getThreadCount() const {
    return threadCount;
}

void WorkerThreadPool::startThreads() {
    running = true;
    for (std::size_t i = 0; i < threadCount; ++i) {
        threads.emplace_back([this] { workerLoop(); });
    }
}

void WorkerThreadPool::stopThreads() {
    running = false;
    conditionQueue.notify_all();
    for (auto& thread : threads) {
        thread.join();
    }
    threads.clear();
}

std::shared_ptr<Job::JobData> WorkerThreadPool::awaitWork() {
    std::unique_lock<std::mutex> lock(mutex);
    while (running) {
        if (jobs.empty()) {
            conditionQueue.wait_for(lock, std::chrono::seconds(1));
        }

        if (running && !jobs.empty()) {
            auto jobData = jobs.front();
            jobs.pop_front();
            return jobData;
        }
    }

    return nullptr;
}

void WorkerThreadPool::workerLoop() {
    while (running) {
        auto jobData = awaitWork();
        if (jobData) {
            {
                std::unique_lock<std::mutex> lock(jobData->mutex);

                // Check/update activity status
                int status = Job::Status::WAITING;
                if (jobData->status.compare_exchange_strong(status, Job::Status::ACTIVE)) {
                    // Do work
                    jobData->func();

                    // Indicate completion status
                    jobData->status = Job::Status::DONE;
                }
            }

            // Notify anyone waiting for job completion
            jobData->condition.notify_all();
        }
    }
}


} // namespace megamol::ImageSeries::util
