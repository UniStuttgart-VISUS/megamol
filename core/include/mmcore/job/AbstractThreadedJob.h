/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/job/AbstractJob.h"
#include "vislib/SmartPtr.h"
#include "vislib/sys/Thread.h"


namespace megamol::core::job {


/**
 * Abstract base class for theaded jobs
 */
class AbstractThreadedJob : public AbstractJob, public vislib::sys::Runnable {
public:
    /**
     * Ctor
     */
    AbstractThreadedJob();

    /**
     * Dtor
     */
    virtual ~AbstractThreadedJob();

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    virtual bool IsRunning() const;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool Start();

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate();

protected:
    /**
     * Answers wether the thread should terminate as soon as possible.
     *
     * @return 'true' if the thread should terminate as soon as possible.
     */
    inline bool shouldTerminate() const {
        return this->terminationRequest;
    }

private:
    /** the job thread */
    vislib::SmartPtr<vislib::sys::Thread> thread;

    /** indicating that the thread should terminate as soon as possible */
    bool terminationRequest;
};

} // namespace megamol::core::job
