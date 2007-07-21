/*
 * testthread.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testthread.h"
#include "testhelper.h"

#include "vislib/Event.h"
#include "vislib/Mutex.h"
#include "vislib/Runnable.h"
#include "vislib/Semaphore.h"
#include "vislib/Thread.h"


using namespace vislib::sys;


// Test runnable
class MyRunnable : public Runnable {

public:

    inline MyRunnable(const UINT id) : cnt(1), id(id) {};

    virtual DWORD Run(void *userData);

private:

    UINT cnt;

    UINT id;
};


DWORD MyRunnable::Run(void *userData) {
    if (userData != NULL) {
        this->cnt = *static_cast<const DWORD *>(userData);
    }

    for (UINT i = 0; i < cnt; i++) {
        //std::cout << "Thread " << this->id << " [" << Thread::CurrentID() 
        //    << "] " << i << std::endl;
        Thread::Sleep(20);
    }

    return cnt;
}


// Test synchronisation.
class SynchronisedRunnable : public Runnable {

public:

    static int Cnt;

    static bool UseSemaphore;

    inline SynchronisedRunnable() {}

    virtual DWORD Run(void *userData);

private:

    static vislib::sys::Mutex mutex;

    static vislib::sys::Semaphore semaphore;

};

DWORD SynchronisedRunnable::Run(void *userData) {
    UINT cnt = (userData != NULL) 
        ? *static_cast<const DWORD *>(userData)
        : 100;

    if (UseSemaphore) {
        std::cout << "Using semaphore ..." << std::endl;
    } else {
        std::cout << "Using mutex ..." << std::endl;
    }

    for (UINT i = 0; i < cnt; i++) {
        if (UseSemaphore) {
            semaphore.Lock();
            SynchronisedRunnable::Cnt++;
            semaphore.Unlock();
        } else {
            mutex.Lock();
            SynchronisedRunnable::Cnt++;
            mutex.Unlock();
        }

        Thread::Sleep(20);
    }

    return cnt;
}

int SynchronisedRunnable::Cnt = 0;
bool SynchronisedRunnable::UseSemaphore = false;
vislib::sys::Mutex SynchronisedRunnable::mutex;
vislib::sys::Semaphore SynchronisedRunnable::semaphore;



// Test Events.
class EventRunnable : public Runnable {

public:
    typedef struct UserData_t {
        UINT cnt;
        vislib::sys::Event *evt;
        bool isSignalDude;
        bool resetEvent;
    } UserData;

    inline EventRunnable() {}

    virtual DWORD Run(void *userData);

private:

    static vislib::sys::Event event;

};

DWORD EventRunnable::Run(void *userData) {
    UserData *ud = static_cast<UserData *>(userData);
    
    if (ud->isSignalDude) {
        std::cout << "Thread " << vislib::sys::Thread::CurrentID() 
            <<  " will signal event " << ud->cnt << " times ..." << std::endl;

        for (UINT i = 0; i < ((ud->resetEvent) ? 1 : 2) * ud->cnt; i++) {
            std::cout << "Event is being signaled." << std::endl;
            ud->evt->Set();

            Thread::Sleep(20);

            if (ud->resetEvent) {
                std::cout << "Event is manually reset." << std::endl;
                ud->evt->Reset();
            }

            Thread::Sleep(20);
        }

    } else {
        std::cout << "Thread " << vislib::sys::Thread::CurrentID()
            << " will wait for being signaled " << ud->cnt << " times ..." 
            << std::endl;

        for (UINT i = 0; i < ud->cnt; i++) {
            ud->evt->Wait();
            std::cout << "Thread " << vislib::sys::Thread::CurrentID()
                << " has been signaled." << std::endl;
        }
    }

    return 0;
}


// Thread test variables.
MyRunnable r1(1);
MyRunnable r2(2);


// Synchronisation test variables.
SynchronisedRunnable s1;
SynchronisedRunnable s2;

// Event test variables.
vislib::sys::Event evtAuto(false);
vislib::sys::Event evtManual(true);
EventRunnable e1;
EventRunnable e2;
EventRunnable e3;

void TestThread(void) {
    DWORD cntLoops = 100;
    
    // Thread tests
    Thread t1(&r1);
    Thread t2(&r2);
    
    ::AssertFalse("Thread 1 is initially not running", t1.IsRunning());
    ::AssertFalse("Thread 2 is initially not running", t2.IsRunning());

    t1.Start(&cntLoops);
    t2.Start(&cntLoops);

    ::AssertTrue("Thread 1 is running", t1.IsRunning());
    ::AssertTrue("Thread 2 is running", t2.IsRunning());

    ::AssertFalse("Thread cannot be started twice", t1.Start());

    t1.Join();
    t2.Join();

    ::AssertEqual("Exit code is number of loops", t1.GetExitCode(), cntLoops);
    ::AssertEqual("Exit code is number of loops", t2.GetExitCode(), cntLoops);


    // Synchronisation tests
    SynchronisedRunnable::Cnt = 0;
    SynchronisedRunnable::UseSemaphore = false;
    Thread t3(&s1);
    Thread t4(&s2);

    t3.Start(&cntLoops);
    t4.Start(&cntLoops);

    t3.Join();
    t4.Join();

    ::AssertEqual("Exit code is number of loops", t3.GetExitCode(), cntLoops);
    ::AssertEqual("Exit code is number of loops", t4.GetExitCode(), cntLoops);
    ::AssertEqual("Counter twice number of loops", DWORD(SynchronisedRunnable::Cnt), 2 * cntLoops);


    SynchronisedRunnable::Cnt = 0;
    SynchronisedRunnable::UseSemaphore = true;
    Thread t5(&s1);
    Thread t6(&s2);

    t5.Start(&cntLoops);
    t6.Start(&cntLoops);

    t5.Join();
    t6.Join();

    ::AssertEqual("Exit code is number of loops", t5.GetExitCode(), cntLoops);
    ::AssertEqual("Exit code is number of loops", t6.GetExitCode(), cntLoops);
    ::AssertEqual("Counter twice number of loops", DWORD(SynchronisedRunnable::Cnt), 2 * cntLoops);


    // Event tests.
    cntLoops = 2;
    EventRunnable::UserData udSignal = { cntLoops, &::evtAuto, true, false };
    EventRunnable::UserData udWait = { cntLoops, &::evtAuto, false, false };

    Thread t7(&e1);
    Thread t8(&e2);
    Thread t9(&e3);

    t7.Start(&udSignal);
    t8.Start(&udWait);
    t9.Start(&udWait);

    t7.Join();
    t8.Join();
    t9.Join();

    udSignal.evt = &evtManual;
    udSignal.resetEvent = true; 
    udWait.evt = &evtManual;
    udWait.resetEvent = true; 

    Thread t10(&e1);
    Thread t11(&e2);
    Thread t12(&e3);

    t10.Start(&udSignal);
    t11.Start(&udWait);
    t12.Start(&udWait);

    t10.Join();
    t11.Join();
    t12.Join();

}
