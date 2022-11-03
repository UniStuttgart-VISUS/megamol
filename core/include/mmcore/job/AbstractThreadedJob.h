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


#ifdef _WIN32
#pragma warning(disable : 4275)
#endif /* _WIN32 */
/**
 * Abstract base class for theaded jobs
 */
class AbstractThreadedJob : public AbstractJob, public vislib::sys::Runnable {
#ifdef _WIN32
#pragma warning(default : 4275)
#endif /* _WIN32 */
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
    virtual bool IsRunning(void) const;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool Start(void);

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate(void);

protected:
    /**
     * Answers wether the thread should terminate as soon as possible.
     *
     * @return 'true' if the thread should terminate as soon as possible.
     */
    inline bool shouldTerminate(void) const {
        return this->terminationRequest;
    }

private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** the job thread */
    vislib::SmartPtr<vislib::sys::Thread> thread;

    /** indicating that the thread should terminate as soon as possible */
    bool terminationRequest;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};

} // namespace megamol::core::job
