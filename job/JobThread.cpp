/*
 * JobThread.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "JobThread.h"
#include "CoreInstance.h"
#include "vislib/sys/Log.h"

using namespace megamol::core;


/*
 * job::JobThread::JobThread
 */
job::JobThread::JobThread() : AbstractThreadedJob(), Module() {
    // intentionally empty ATM
}


/*
 * job::JobThread::~JobThread
 */
job::JobThread::~JobThread() {
    Module::Release();
}


/*
 * job::JobThread::create
 */
bool job::JobThread::create(void) {
    return true; // intentionally empty ATM
}


/*
 * job::JobThread::release
 */
void job::JobThread::release(void) {
    // intentionally empty ATM
}


/*
 * job::JobThread::Run
 */
DWORD job::JobThread::Run(void *userData) {

    // this->GetCoreInstance()->Log().WriteMsg(vislib::sys::Log::LEVEL_INFO,
    //     "JobThread started (CoreLog)");
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "JobThread started (DefaultLog)");

    // TODO: Implement

    const unsigned int allTime = 10000;
    const unsigned int sleepTime = 100;
    const unsigned int iterations = allTime / sleepTime;
    for (unsigned int i = 0; (i < iterations) && !this->shouldTerminate(); i++) {
        vislib::sys::Thread::Sleep(sleepTime);
    }

    // this->GetCoreInstance()->Log().WriteMsg(vislib::sys::Log::LEVEL_INFO,
    //     "JobThread finished (CoreLog)");
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "JobThread finished (DefaultLog)");

    this->signalEnd(this->shouldTerminate());
    return 0;
}
