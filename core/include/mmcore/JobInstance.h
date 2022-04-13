/*
 * JobInstance.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_JOBINSTANCE_H_INCLUDED
#define MEGAMOLCORE_JOBINSTANCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/ModuleNamespace.h"
#include "mmcore/job/AbstractJob.h"
#include "vislib/forceinline.h"


namespace megamol {
namespace core {


/**
 * class of job instances
 */
class JobInstance : public ModuleNamespace {
public:
    /**
     * Ctor.
     */
    JobInstance(void);

    /**
     * Dtor.
     */
    virtual ~JobInstance(void);

    /**
     * Initializes the view instance.
     *
     * @param ns The namespace object to be replaced.
     * @param view The view module to be used.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool Initialize(ModuleNamespace::ptr_type ns, job::AbstractJob* job);

    /**
     * Gets the view object encapsuled by this instance.
     *
     * @return The view object.
     */
    VISLIB_FORCEINLINE job::AbstractJob* Job(void) {
        return this->job;
    }

    /**
     * Signals the job that it should be terminated as soon as possible.
     * The module must not be immediatly removed from the module graph.
     */
    void Terminate(void);

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    virtual void ClearCleanupMark(void);

    /**
     * Performs the cleanup operation by removing and deleteing of all
     * marked objects.
     */
    virtual void PerformCleanup(void);

private:
    /** The job controller module */
    job::AbstractJob* job;
};


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_JOBINSTANCE_H_INCLUDED */
