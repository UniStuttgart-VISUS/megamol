/*
 * testthreadpool.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testthreadpool.h"

#include "vislib/ThreadPool.h"
#include "testhelper.h"
#include <climits>


using namespace vislib::sys;

// Ensure that a whole line is output at once on cout.
static CriticalSection COUT_IO_LOCK;
#define LOCK_COUT ::COUT_IO_LOCK.Lock()
#define UNLOCK_COUT ::COUT_IO_LOCK.Unlock()


class Horcher : public vislib::sys::ThreadPoolListener {

public:

    inline Horcher(void) : ThreadPoolListener(), cntAborted(0), cntCompleted(0) {};

    void OnUserWorkItemAborted(ThreadPool& src, Runnable *runnable,
        void *userData) throw();

    void OnUserWorkItemAborted(ThreadPool& src, Runnable::Function runnable,
        void *userData) throw();

    void OnUserWorkItemCompleted(ThreadPool& src, Runnable *runnable, 
        void *userData, const DWORD exitCode) throw();

    void OnUserWorkItemCompleted(ThreadPool& src, Runnable::Function runnable,
        void *userData, const DWORD exitCode) throw();

    UINT cntAborted;
    UINT cntCompleted;

};

void Horcher::OnUserWorkItemAborted(ThreadPool& src, Runnable *runnable,
        void *userData) throw() {
    this->cntAborted++;
}

void Horcher::OnUserWorkItemAborted(ThreadPool& src, 
        Runnable::Function runnable, void *userData) throw() {
    this->cntAborted++;
}

void Horcher::OnUserWorkItemCompleted(ThreadPool& src, Runnable *runnable,
        void *userData, const DWORD exitCode) throw() {
    this->cntCompleted++;
}

void Horcher::OnUserWorkItemCompleted(ThreadPool& src, 
        Runnable::Function runnable, void *userData, 
        const DWORD exitCode) throw() {
    this->cntCompleted++;
}



class Doweler : public vislib::sys::Runnable {
    
public:

    inline Doweler(void) : Runnable() {};

    virtual DWORD Run(void *userData);

    virtual bool Terminate(void);
};


DWORD Doweler::Run(void *userData) {
    UINT_PTR dowel = reinterpret_cast<UINT_PTR>(userData);

    for (int i = 0; i < 5; i++) {
        LOCK_COUT;
        std::cout << "Doweler " << dowel << " is doweling ..." << std::endl;
        UNLOCK_COUT;
        Thread::Sleep(2);
    }
    return 0;
}

bool Doweler::Terminate(void) {
    return true;
}



class Crowbarer : public Runnable {
public:

    inline Crowbarer(void) : Runnable() {};

    virtual DWORD Run(void *userData);

    virtual bool Terminate(void);

    static Semaphore sem;
};


DWORD Crowbarer::Run(void *userData) {
    UINT_PTR crowbar = reinterpret_cast<UINT_PTR>(userData);

    sem.Lock();
    LOCK_COUT;
    std::cout << "Crowbarer " << crowbar << " is crowbaring ..." << std::endl;
    UNLOCK_COUT;
    return 0;
}

bool Crowbarer::Terminate(void) {
    return true;
}

Semaphore Crowbarer::sem(0l, LONG_MAX);



void TestThreadPool(void) {

    const int CNT_THREADS = 2;
    const int CNT_DOWELERS = 10;
    const int CNT_CROWBARERS = 10;
    const int CNT_UNLOCKED_CROWBARERS = 3;
    ThreadPool pool;
    Horcher a6;
    Doweler dowelers[CNT_DOWELERS];
    Crowbarer crowbarers[CNT_CROWBARERS];

    pool.AddListener(&a6);

    ::AssertEqual("No threads initially.", pool.GetTotalThreads(), SIZE_T(0));
    ::AssertEqual("No active threads initially.", pool.GetActiveThreads(), SIZE_T(0));
    ::AssertEqual("No idle threads initially.", pool.GetAvailableThreads(), SIZE_T(0));

    for (INT_PTR i = 0; i < CNT_DOWELERS; i++) {
        pool.QueueUserWorkItem(&dowelers[i], reinterpret_cast<void *>(i), false);
    }
    pool.SetThreadCount(CNT_THREADS);

    pool.Wait();
    ::AssertTrue("Idle threads exist after wait.", pool.GetAvailableThreads() > 0);
    ::AssertEqual("No active threads after wait.", pool.GetActiveThreads(), SIZE_T(0));

    ::AssertEqual("Nothing aborted.", a6.cntAborted, UINT(0));
    ::AssertEqual("Everything completed.", a6.cntCompleted, UINT(CNT_DOWELERS));

    a6.cntAborted = 0;
    a6.cntCompleted = 0;
    for (INT_PTR i = 0; i < CNT_CROWBARERS; i++) {
        pool.QueueUserWorkItem(crowbarers + i, reinterpret_cast<void *>(i), false);
    }

    for (int i = 0; i < CNT_UNLOCKED_CROWBARERS; i++) {
        Crowbarer::sem.Unlock();
    }
    pool.AbortPendingUserWorkItems();

    for (int i = 0; i < CNT_CROWBARERS; i++) {
        Crowbarer::sem.Unlock();
    }

    pool.Terminate();
    ::AssertEqual("No threads after terminate.", pool.GetTotalThreads(), SIZE_T(0));
    ::AssertEqual("No active threads after terminate.", pool.GetActiveThreads(), SIZE_T(0));
    ::AssertEqual("No idle threads after terminate.", pool.GetAvailableThreads(), SIZE_T(0));

    ::AssertTrue("Work items have been completed.", a6.cntCompleted > 0);
    ::AssertTrue("Work items have been aborted.", a6.cntAborted > 0);

    pool.RemoveListener(&a6);
}
