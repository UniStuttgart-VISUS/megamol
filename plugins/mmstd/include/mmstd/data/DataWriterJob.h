/*
 * DataWriterJob.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DATAWRITERJOB_H_INCLUDED
#define MEGAMOLCORE_DATAWRITERJOB_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/job/AbstractThreadedJob.h"


namespace megamol {
namespace core {
namespace job {


/**
 * Class implementing a simple thread for the job.
 */
class DataWriterJob : public AbstractThreadedJob, public Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "DataWriterJob";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Threaded job conrtolling a data writer";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) {
        return false;
    }

    /**
     * Ctor
     */
    DataWriterJob();

    /**
     * Dtor
     */
    virtual ~DataWriterJob();

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    /**
     * Perform the work of a thread.
     *
     * @param userData A pointer to user data that are passed to the thread,
     *                 if it started.
     *
     * @return The application dependent return code of the thread. This
     *         must not be STILL_ACTIVE (259).
     */
    virtual DWORD Run(void* userData);

    /** Slot connecting to the controlled writer */
    CallerSlot writerSlot;

    /** Flag if the writer is abortable */
    bool abortable;
};


} /* end namespace job */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DATAWRITERJOB_H_INCLUDED */
