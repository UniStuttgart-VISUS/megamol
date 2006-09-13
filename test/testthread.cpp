/*
 * testthread.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testthread.h"
#include "testhelper.h"

#include "vislib/Runnable.h"
#include "vislib/Thread.h"


using namespace vislib::sys;


class MyRunnable : public Runnable {

public:

    inline MyRunnable(const UINT id) : cnt(1), id(id) {};

    virtual DWORD Run(const void *userData);

private:

    UINT cnt;

    UINT id;
};


DWORD MyRunnable::Run(const void *userData) {
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


MyRunnable r1(1);
MyRunnable r2(2);

void TestThread(void) {
    Thread t1(&r1);
    Thread t2(&r2);
    DWORD cntLoops = 100;

    t1.Start(&cntLoops);
    t2.Start(&cntLoops);

    ::AssertTrue("Thread 1 is running", t1.IsRunning());
    ::AssertTrue("Thread 2 is running", t2.IsRunning());

    ::AssertFalse("Thread cannot be started twice", t1.Start());

    t1.Join();
    t2.Join();

    ::AssertEqual("Exit code is number of loops", t1.GetExitCode(), cntLoops);
    ::AssertEqual("Exit code is number of loops", t2.GetExitCode(), cntLoops);
}
