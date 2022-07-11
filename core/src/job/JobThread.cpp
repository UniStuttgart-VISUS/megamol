/*
 * JobThread.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/job/JobThread.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/log/Log.h"

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
DWORD job::JobThread::Run(void* userData) {

    // this->GetCoreInstance()->Log().WriteInfo(
    //     "JobThread started (CoreLog)");
    megamol::core::utility::log::Log::DefaultLog.WriteInfo( "JobThread started (DefaultLog)");

    // TODO: Implement

    const unsigned int allTime = 10000;
    const unsigned int sleepTime = 100;
    const unsigned int iterations = allTime / sleepTime;
    for (unsigned int i = 0; (i < iterations) && !this->shouldTerminate(); i++) {
        vislib::sys::Thread::Sleep(sleepTime);
    }

    // this->GetCoreInstance()->Log().WriteInfo(
    //     "JobThread finished (CoreLog)");
    megamol::core::utility::log::Log::DefaultLog.WriteInfo( "JobThread finished (DefaultLog)");

    this->signalEnd(this->shouldTerminate());
    return 0;
}
