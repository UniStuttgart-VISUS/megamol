/*
 * AbstractJob.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTJOB_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTJOB_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/AbstractParam.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/assert.h"


namespace megamol {
namespace core {
namespace job {


/**
 * Abstract base class of processing jobs
 */
class AbstractJob {
public:
    /** Ctor. */
    AbstractJob(void);

    /** Dtor. */
    virtual ~AbstractJob(void);

    /**
     * Answers whether the given parameter is relevant for this job.
     *
     * @param param The parameter to test.
     *
     * @return 'true' if 'param' is relevant, 'false' otherwise.
     */
    virtual bool IsParamRelevant(const vislib::SmartPtr<param::AbstractParam>& param) const;

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    virtual bool IsRunning(void) const = 0;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool Start(void) = 0;

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate(void) = 0;

protected:
    /**
     * Signals the application that the job has been started.
     */
    void signalStart(void);

    /**
     * Signals the application that the job has ended.
     *
     * @param wasTerminated May indicate that the job was terminated,
     *                      instead of finished.
     */
    void signalEnd(bool wasTerminated = false);

private:
};


} /* end namespace job */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTJOB_H_INCLUDED */
