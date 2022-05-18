/*
 * AbstractThreadedJob.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/job/AbstractThreadedJob.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"

using namespace megamol::core;


/*
 * job::AbstractThreadedJob::AbstractThreadedJob
 */
job::AbstractThreadedJob::AbstractThreadedJob()
        : AbstractJob()
        , vislib::sys::Runnable()
        , thread(NULL)
        , terminationRequest(false) {
    // intentionally empty
}


/*
 * job::AbstractThreadedJob::~AbstractThreadedJob
 */
job::AbstractThreadedJob::~AbstractThreadedJob() {
    if (!this->thread.IsNull()) {
        if (this->thread->IsRunning()) {
            this->thread->Terminate(false);
        }
    }
}


/*
 * job::AbstractThreadedJob::IsRunning
 */
bool job::AbstractThreadedJob::IsRunning(void) const {
    return !this->thread.IsNull() && this->thread->IsRunning();
}


/*
 * job::AbstractThreadedJob::Start
 */
bool job::AbstractThreadedJob::Start(void) {
    try {
        this->terminationRequest = false;
        this->thread = new vislib::sys::Thread(this);
        if (this->thread->Start()) {
            this->signalStart();
            return true;
        }
    } catch (...) {}
    return false;
}


/*
 * job::AbstractThreadedJob::Terminate
 */
bool job::AbstractThreadedJob::Terminate(void) {
    this->terminationRequest = true;
    return true;
}
