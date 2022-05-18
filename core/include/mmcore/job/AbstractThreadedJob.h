/*
 * AbstractThreadedJob.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTTHREADEDJOB_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTTHREADEDJOB_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/job/AbstractJob.h"
#include "mmcore/utility/sys/Thread.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {
namespace job {


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


} /* end namespace job */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTTHREADEDJOB_H_INCLUDED */
