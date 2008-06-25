/*
 * ThreadPool.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_THREADPOOL_H_INCLUDED
#define VISLIB_THREADPOOL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CriticalSection.h"
#include "vislib/Event.h"
#include "vislib/RunnableThread.h"
#include "vislib/Semaphore.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/types.h"
#include "vislib/WorkItemCompletedListener.h"


namespace vislib {
namespace sys {


    /**
     * TODO: comment class
     */
    class ThreadPool {

    public:

        /** Ctor. */
        ThreadPool(void);

        /** Dtor. */
        ~ThreadPool(void);

        SIZE_T CountUserWorkItems(void) const;

        void QueueUserWorkItem(Runnable *runnable, void *userData = NULL, 
            WorkItemCompletedListener *workItemCompletedListener = NULL,
            const bool noDefaultThreads = false);

        void SetThreadCount(const SIZE_T threadCount);

        void Terminate(void);

        inline bool Wait(const DWORD timeout = Event::TIMEOUT_INFINITE) {
            return this->evtAllCompleted.Wait(timeout);
        }


    private:

        class Worker : public Runnable {

        public:

            Worker(void);

            virtual ~Worker(void);

            /**
             * Perform the work of a thread.
             *
             * @param pool Pointer to the thread pool that the worker is 
             *             working for. The caller remains owner of the memory
             *             designated by 'pool' and must ensure it lives as long
             *             as the worker..
             *
             * @return The application dependent return code of the thread. This 
             *         must not be STILL_ACTIVE (259).
             */
            virtual DWORD Run(void *pool);

        private:

            /** The pool to get the work items from. */
            ThreadPool *pool;
        };

        typedef struct WorkItem_t {
            WorkItemCompletedListener *listener;
            Runnable *runnable;
            void *userData;

            inline bool operator ==(const struct WorkItem_t& rhs) const {
                return ((this->listener == rhs.listener)
                    && (this->runnable == rhs.runnable)
                    &&(this->userData == rhs.userData));
            }
        } WorkItem;

        SIZE_T cntActiveThreads;

        SIZE_T cntThreads;

        Event evtAllCompleted;

        mutable CriticalSection lockCntActiveThreads;

        /** Semaphore for waiting while 'workItems' is empty. */
        Semaphore semBlockWorker;

        /** Lock for protecting 'workItems'. */
        mutable CriticalSection lockWorkItems;

        /** The list of pending work items. */
        SingleLinkedList<WorkItem> workItems;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_THREADPOOL_H_INCLUDED */

