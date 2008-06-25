/*
 * ThreadPool.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ThreadPool.h"

#include <climits>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SystemInformation.h"
#include "vislib/Trace.h"


/*
 * vislib::sys::ThreadPool::ThreadPool
 */
vislib::sys::ThreadPool::ThreadPool(void) : cntActiveThreads(0), 
        cntThreads(0), evtAllCompleted(true), semBlockWorker(0l, LONG_MAX) {
    // Nothing to do.
}


/*
 * vislib::sys::ThreadPool::~ThreadPool
 */
vislib::sys::ThreadPool::~ThreadPool(void) {
    // TODO
    //this->SetThreadCount(0);    // Blocks!!!
}


/*
 * vislib::sys::ThreadPool::CountUserWorkItems
 */
SIZE_T vislib::sys::ThreadPool::CountUserWorkItems(void) const {
    this->lockWorkItems.Lock();
    SIZE_T retval = this->workItems.Count();
    this->lockWorkItems.Unlock();
    return retval;
}


/*
 * vislib::sys::ThreadPool::QueueUserWorkItem
 */
void vislib::sys::ThreadPool::QueueUserWorkItem(Runnable *runnable, 
        void *userData, WorkItemCompletedListener *workItemCompletedListener,
        const bool noDefaultThreads) {
    WorkItem workItem;
    workItem.listener = workItemCompletedListener;
    workItem.runnable = runnable;
    workItem.userData = userData;

    this->lockWorkItems.Lock();
    this->workItems.Append(workItem);
    this->evtAllCompleted.Reset();          // There must be working threads.
    this->lockWorkItems.Unlock();
    this->semBlockWorker.Unlock();      // Signal semaphore.

    if (!noDefaultThreads && (this->cntThreads < 1)) {
        this->SetThreadCount(SystemInformation::ProcessorCount());
    }
}


/*
 * vislib::sys::ThreadPool::SetThreadCount
 */
void vislib::sys::ThreadPool::SetThreadCount(const SIZE_T threadCount) {
    if (threadCount < this->cntThreads) {
        throw IllegalParamException("The number of threads in the thread pool "
            "cannot be reduced.", __FILE__, __LINE__);
    }

    for (; this->cntThreads < threadCount; this->cntThreads++) {
        (new RunnableThread<Worker>())->Start(this);
    }

    
}


/*
 * vislib::sys::ThreadPool::Terminate
 */
void vislib::sys::ThreadPool::Terminate(void) {
    this->lockWorkItems.Lock();
    this->workItems.Clear();
    BECAUSE_I_KNOW(this->cntActiveThreads == 0);
    this->evtAllCompleted.Reset();
    for (SIZE_T i = 0; i < this->cntThreads; i++) {
        this->semBlockWorker.Unlock();
    }
    this->lockWorkItems.Unlock();
    this->evtAllCompleted.Wait();
}


/*
 * vislib::sys::ThreadPool::Worker::Worker
 */
vislib::sys::ThreadPool::Worker::Worker(void) : Runnable(), pool(NULL) {
    // Nothing to do.
}


/*
 * vislib::sys::ThreadPool::Worker::~Worker
 */
vislib::sys::ThreadPool::Worker::~Worker(void) {
    // Nothing to do.
}


/*
 * DWORD vislib::sys::ThreadPool::Worker::Run
 */
DWORD vislib::sys::ThreadPool::Worker::Run(void *pool) {
    ASSERT(pool != NULL);
    this->pool = static_cast<ThreadPool *>(pool);

    TRACE(Trace::LEVEL_VL_INFO, "Worker thread [%u] started.\n", 
        Thread::CurrentID());

    while (true) {
        TRACE(Trace::LEVEL_VL_INFO, "ThreadPool thread [%u] is waiting for "
            "work ...\n", Thread::CurrentID());
        this->pool->semBlockWorker.Lock();

        this->pool->lockWorkItems.Lock();
        this->pool->lockCntActiveThreads.Lock();

        if (this->pool->workItems.IsEmpty()) {
            /* 
             * If the thread is woken while the queue is empty, this is the 
             * signal to exit.
             */
            if (++this->pool->cntActiveThreads == this->pool->cntThreads) {
                this->pool->evtAllCompleted.Set();
            }
            this->pool->lockCntActiveThreads.Unlock();
            this->pool->lockWorkItems.Unlock();
            delete this;
            return 0;
        }

        this->pool->cntActiveThreads++;
        WorkItem workItem = this->pool->workItems.First();
        this->pool->workItems.RemoveFirst();

        this->pool->lockCntActiveThreads.Unlock();
        this->pool->lockWorkItems.Unlock();

        TRACE(Trace::LEVEL_VL_INFO, "ThreadPool thread [%u] is working ...\n",
            Thread::CurrentID());
        DWORD exitCode = workItem.runnable->Run(workItem.userData);
        TRACE(Trace::LEVEL_VL_INFO, "ThreadPool thread [%u] completed work "
            "item with exit code %u\n", Thread::CurrentID(), exitCode);

        if (workItem.listener != NULL) {
            workItem.listener->OnWorkItemCompleted(workItem.runnable,
                workItem.userData, exitCode);
        }

        this->pool->lockWorkItems.Lock();
        this->pool->lockCntActiveThreads.Lock();
        if ((--this->pool->cntActiveThreads == 0) 
                && this->pool->workItems.IsEmpty()) {
            this->pool->evtAllCompleted.Set();
        }
        this->pool->lockCntActiveThreads.Unlock();
        this->pool->lockWorkItems.Unlock();
    }
}
