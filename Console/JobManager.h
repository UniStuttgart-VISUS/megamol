/*
 * JobManager.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_JOBMANAGER_H_INCLUDED
#define MEGAMOLCON_JOBMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "CoreHandle.h"


namespace megamol {
namespace console {

    class JobManager{
    public:

        /**
         * The singelton instance method.
         *
         * @return The only instance of this class.
         */
        static JobManager* Instance(void);

        /** Dtor. */
        ~JobManager(void);

        /**
         * Adds a job to the manager.
         *
         * @param job to be added.
         */
        void Add(vislib::SmartPtr<CoreHandle>& job);

        /**
         * Checks all running jobs.
         *
         * @return 'true' if there is at least on job still running,
         *         'false' if there are no more running jobs.
         */
        bool CheckJobs(void);

        /**
         * Tries to terminate all jobs still running.
         */
        void TerminateJobs(void);

    private:

        /** Private ctor. */
        JobManager(void);

        /** The viewing windows. */
        vislib::SingleLinkedList<vislib::SmartPtr<CoreHandle> > jobs;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_JOBMANAGER_H_INCLUDED */
