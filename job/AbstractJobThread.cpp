/*
 * AbstractJobThread.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractJobThread.h"
#include "vislib/Log.h"

using namespace megamol::core;


/*
 * job::AbstractJobThread::AbstractJobThread
 */
job::AbstractJobThread::AbstractJobThread() : AbstractJob(),
        vislib::sys::Runnable(), thread(NULL), terminationRequest(false) {
    // intentionally empty
}


/*
 * job::AbstractJobThread::~AbstractJobThread
 */
job::AbstractJobThread::~AbstractJobThread() {
    if (!this->thread.IsNull()) {
        if (this->thread->IsRunning()) {
            this->thread->Terminate(false);
        }
    }
}


/*
 * job::AbstractJobThread::IsRunning
 */
bool job::AbstractJobThread::IsRunning(void) const {
    return !this->thread.IsNull() && this->thread->IsRunning();
}


/*
 * job::AbstractJobThread::Start
 */
bool job::AbstractJobThread::Start(void) {
    try {
        this->terminationRequest = false;
        this->thread = new vislib::sys::Thread(this);
        if (this->thread->Start()) {
            this->signalStart();
            return true;
        }
    } catch(...) {
    }
    return false;
}


/*
 * job::AbstractJobThread::Terminate
 */
bool job::AbstractJobThread::Terminate(void) {
    this->terminationRequest = true;
    return true;
}
