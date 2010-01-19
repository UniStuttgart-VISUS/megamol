/*
 * AbstractJobThread.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTJOBTHREAD_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTJOBTHREAD_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "job/AbstractJob.h"
#include "vislib/SmartPtr.h"
#include "vislib/Thread.h"


namespace megamol {
namespace core {
namespace job {


    /**
     * Abstract base class for theaded jobs
     */
    class AbstractJobThread : public AbstractJob,
        public vislib::sys::Runnable {
    public:

        /**
         * Ctor
         */
        AbstractJobThread();

        /**
         * Dtor
         */
        virtual ~AbstractJobThread();

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

        /** the job thread */
        vislib::SmartPtr<vislib::sys::Thread> thread;

        /** indicating that the thread should terminate as soon as possible */
        bool terminationRequest;

    };


} /* end namespace job */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTJOBTHREAD_H_INCLUDED */
