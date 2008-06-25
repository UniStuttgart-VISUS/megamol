/*
 * testthreadpool.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testthreadpool.h"

#include "vislib/ThreadPool.h"
#include "testhelper.h"


// Ensure that a whole line is output at once on cout.
static vislib::sys::CriticalSection COUT_IO_LOCK;
#define LOCK_COUT ::COUT_IO_LOCK.Lock()
#define UNLOCK_COUT ::COUT_IO_LOCK.Unlock()


class Doweler : public vislib::sys::Runnable {
    
public:

    inline Doweler(void) : Runnable() {};

    virtual DWORD Run(void *userData);

    virtual bool Terminate(void);
};


DWORD Doweler::Run(void *userData) {
    int dowel = reinterpret_cast<int>(userData);

    for (int i = 0; i < 5; i++) {
        LOCK_COUT;
        std::cout << dowel << " doweling ..." << std::endl;
        UNLOCK_COUT;
        vislib::sys::Thread::Sleep(2);
    }
    return 0;
}

bool Doweler::Terminate(void) {
    return true;
}



void TestThreadPool(void) {
    using namespace vislib::sys;

    const int CNT_DOWELERS = 10;
    ThreadPool pool;
    Doweler dowelers[CNT_DOWELERS];

    for (int i = 0; i < 10; i++) {
        pool.QueueUserWorkItem(&dowelers[i], reinterpret_cast<void *>(i));
    }

    pool.Wait();
    pool.Terminate();
}