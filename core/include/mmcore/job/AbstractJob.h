/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/param/AbstractParam.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/assert.h"


namespace megamol::core::job {


/**
 * Abstract base class of processing jobs
 */
class AbstractJob {
public:
    /** Ctor. */
    AbstractJob();

    /** Dtor. */
    virtual ~AbstractJob();

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    virtual bool IsRunning() const = 0;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool Start() = 0;

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate() = 0;

protected:
    /**
     * Signals the application that the job has been started.
     */
    void signalStart();

    /**
     * Signals the application that the job has ended.
     *
     * @param wasTerminated May indicate that the job was terminated,
     *                      instead of finished.
     */
    void signalEnd(bool wasTerminated = false);

private:
};


} // namespace megamol::core::job
