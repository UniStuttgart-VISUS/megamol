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
#include "vislib/ThreadPoolListener.h"


namespace vislib {
namespace sys {


    /**
     * This thread pool is able to run an arbitrary number of classes 
     * implementing the Runnables interface reusing a certain number of threads.
     *
     * Note: The Runnables are passed as pointers into the pool. The pool does 
     * not create deep copies, but only stores the pointers to the Runnables and
     * their startup parameters. The user is responsible for ensuring that the
     * Runnable and the data designated by the user data pointer exist until the
     * Runnable has been executed in one of the thread pool thread.
     * This can be realised by the following pattern: Allocate the Runnable and
     * the input parameters on the heap and pass the pointers into the pool. 
     * Register a ThreadPoolListener on the pool and wait for the Runnable to
     * be completed or aborted. The listener will receive the pointer to the
     * Runnable and to the input data in both cases. Release the memory in the
     * event handling methods.
     */
    class ThreadPool {

    public:

        /** Ctor. */
        ThreadPool(void);

        /** Dtor. */
        ~ThreadPool(void);

        /**
         * Remove all pending user work items from the queue.
         *
         * @return The number of items actually removed.
         */
        SIZE_T AbortPendingUserWorkItems(void);

        /**
         * Add a new ThreadPoolListener to be informed about user work items 
         * being completed or aborted. The caller remains owner of the memory 
         * designated by 'listener' and must ensure that the object exists as 
         * long as the listener is registered.
         *
         * @param listener The listener to register. This must not be NULL.
         */
        void AddListener(ThreadPoolListener *listener);

        /**
         * Answer the number of threads currently working.
         *
         * @return The number of threads currently working.
         */
        SIZE_T GetActiveThreads(void) const;

        /**
         * Answer the number of threads currently idling.
         *
         * @return The number of threads currently idling.
         */
        SIZE_T GetAvailableThreads(void) const;

        /**
         * Answer the total number of threads available in the pool.
         *
         * @return The total number of threads in the pool.
         */
        SIZE_T GetTotalThreads(void) const;

        /**
         * Answer the number of work items which are currently in the queue.
         *
         * @return The number of work items currently in the queue.
         */
        SIZE_T CountUserWorkItems(void) const;

        /**
         * Queue a new work item for execution in a pool thread.
         *
         * This method can only be called until the Terminate() method for 
         * shutting down the pool was called and closed the queue.
         *
         * The caller is resposible for that 'runnable' and 'userData' exists
         * until they have been processed by a pool thread. See the class 
         * documentation on how to ensure this.
         *
         * @param runnable             The Runnable that does the work. This 
         *                             must not be a NULL pointer.
         * @param userData             This pointer is passed to the Runnable
         *                             once it is executed.
         * @param createDefaultThreads If set true, the pool creates threads if
         *                             no threads already exist. The default 
         *                             number of threads is one for each 
         *                             available processor.
         *
         * @throws IllegalStateException If the work item queue has been closed.
         * @throws IllegalParamException If 'runnable' is NULL.
         */
        void QueueUserWorkItem(Runnable *runnable, void *userData = NULL, 
            const bool createDefaultThreads = true);

        /**
         * Queue a new work item for execution in a pool thread.
         *
         * This method can only be called until the Terminate() method for 
         * shutting down the pool was called and closed the queue.
         *
         * The caller is resposible for that 'runnable' and 'userData' exists
         * until they have been processed by a pool thread. See the class 
         * documentation on how to ensure this.
         *
         * @param runnable             The Runnable::Function that does the work. 
         *                             This must not be a NULL pointer.
         * @param userData             This pointer is passed to the Runnable
         *                             once it is executed.
         * @param createDefaultThreads If set true, the pool creates threads if
         *                             no threads already exist. The default 
         *                             number of threads is one for each 
         *                             available processor.
         *
         * @throws IllegalStateException If the work item queue has been closed.
         * @throws IllegalParamException If 'runnable' is NULL.
         */
        void QueueUserWorkItem(Runnable::Function runnable, 
            void *userData = NULL, const bool createDefaultThreads = true);

        /**
         * Removes, if registered, 'listener' from the list of objects informed
         * about thread pool events. The caller remains owner of the memory 
         * designated by 'listener'.
         *
         * @param listener The listener to be removed. Nothing happens, if the
         *                 listener was not registered.
         */
        void RemoveListener(ThreadPoolListener *listener);

        /**
         * Set the number of worker threads to use.
         *
         * The number of threads cannot be reduced, i. e. 'threadCount' must be
         * at least this->GetTotalThreads().
         *
         * @param threadCount The number of threads to use.
         *
         * @throws IllegalParamException If 'threadCount' is too small.
         */
        void SetThreadCount(const SIZE_T threadCount);

        /**
         * Wait for all queued or running work items to be completed and exit 
         * all worker threads afterwards.
         *
         * @param abortPending If true, all items currently in the queue are
         *                     removed and only the active items are completed. 
         *                     Otherwise, the method returns once all queued 
         *                     items have been completed.
         */
        void Terminate(const bool abortPending = false);

        /**
         * Wait for all queued work items to be completed.
         *
         * @param timeout A timeout for waiting. Defaults to TIMEOUT_INFINITE.
         *
         * @return true If the operation completed successfully, 
         *         false if a timeout occurred.
         */
        inline bool Wait(const DWORD timeout = Event::TIMEOUT_INFINITE) {
            return this->evtAllCompleted.Wait(timeout);
        }

    private:

        /**
         * This is the worker Runnable that executes other Runnables that
         * have been queued as user work items. The runnable runs until the 
         * 'semBlockWorker' member of 'pool' is in signaled state, but the work
         * item queue is empty at the same time.
         */
        class Worker : public Runnable {

        public:

            /** Ctor. */
            Worker(void);

            /** Dtor. */
            virtual ~Worker(void);

            /**
             * Perform the work of a thread.
             *
             * @param pool Pointer to the thread pool that the worker is 
             *             working for. The caller remains owner of the memory
             *             designated by 'pool' and must ensure it lives as long
             *             as the worker..
             *
             * @return 0, always.
             */
            virtual DWORD Run(void *pool);

        private:

            /** The pool to get the work items from. */
            ThreadPool *pool;
        }; /* end class Worker */

        /** Used to store the work items and their input data. */
        typedef struct WorkItem_t {
            Runnable *runnable;
            Runnable::Function runnableFunction;
            void *userData;

            inline bool operator ==(const struct WorkItem_t& rhs) const {
                return ((this->runnable == rhs.runnable)
                    && (this->runnableFunction == rhs.runnableFunction)
                    &&(this->userData == rhs.userData));
            }
        } WorkItem;

        /* 
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        ThreadPool(const ThreadPool& rhs);

        /**
         * Fire the abort event. 
         *
         * This method is thread-safe
         *
         * @param workItem The work item that was aborted.
         * @param userData The user input associated with the work item.
         */
        void fireUserWorkItemAborted(WorkItem& workItem);

        /**
         * Fire the completed event. 
         *
         * This method is thread-safe
         *
         * @param workItem The work item that was completed.
         * @param exitCode The exit code of the work item.
         */
        void fireUserWorkItemCompleted(WorkItem& workItem, 
            const DWORD exitCode);

        /**
         * Queue a new work item for execution in a pool thread.
         *
         * This method can only be called until the Terminate() method for 
         * shutting down the pool was called and closed the queue.
         *
         * @param workItem             The work item to be queued.
         * @param createDefaultThreads If set true, the pool creates threads if
         *                             no threads already exist. The default 
         *                             number of threads is one for each 
         *                             available processor.
         * 
         * @throws IllegalStateException If the work item queue has been closed.
         * @throws IllegalParamException If both, the 'runnable' and the 
         *                               'runnnableFunction' in the 'workItem' 
         *                               are both NULL or not NULL.
         */
        void queueUserWorkItem(WorkItem& workItem, 
            const bool createDefaultThreads);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         *
         * @throws IllegalParameException if (this != &rhs).
         */
        ThreadPool& operator =(const ThreadPool& rhs);

        /** The number of threads currently working on some work item. */
        SIZE_T cntActiveThreads;

        /** The total number of threads (active and idling). */
        SIZE_T cntTotalThreads;

        /** 
         * This event is in signaled state while no work item is pending or
         * being processed.
         */
        Event evtAllCompleted;

        /** 
         * This flag determines whether it is possible to add new work items
         * to the queue. If the thread pool is being shut down, the queue is
         * closed by this flag. 
         * When accessing this attribute, 'lockQueue' must be held.
         */
        bool isQueueOpen;

        /** 
         * The list of observers to be notified about completed or aborted
         * user work items.
         * When accessing this attribute, 'lockListeners' must be held.
         */
        SingleLinkedList<ThreadPoolListener *> listeners;

        /** Protects access to 'listeners'. */
        CriticalSection lockListeners;

        /** Lock for protecting 'queue' and 'isQueueOpen'. */
        mutable CriticalSection lockQueue;

        /**
         * Critical section for protecting 'cntActiveThreads' and 
         * 'cntTotalThreads'. The lock must be hold if any of the counters
         * is accessed to ensure a consistent view of the total number of
         * threads.
         */
        mutable CriticalSection lockThreadCounters;

        /** 
         * The list of pending work items. 
         * When accessing this attribute, 'lockQueue' must be held.
         */
        SingleLinkedList<WorkItem> queue;

        /** Semaphore for waiting while 'queue' is empty. */
        Semaphore semBlockWorker;
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_THREADPOOL_H_INCLUDED */
