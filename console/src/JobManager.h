/*
 * JobManager.h
 *
 * Copyright (C) 2008, 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_JOBMANAGER_H_INCLUDED
#define MEGAMOLCON_JOBMANAGER_H_INCLUDED
#pragma once

#include <vector>
#include <memory>
#include "CoreHandle.h"


namespace megamol {
namespace console {

    class JobManager {
    public:

        /**
         * The singelton instance method.
         *
         * @return The only instance of this class.
         */
        static JobManager& Instance(void);

        /** Dtor. */
        ~JobManager(void);

        bool IsAlive(void) const;
        void Update(bool force = false);
        void Shutdown(void);

        bool InstantiatePendingJob(void *hCore);

    private:

        class Job {
        public:
            Job();
            ~Job();
            bool IsRunning();
            void Start();

            CoreHandle hJob;
            bool started;
        };

        /** Private ctor. */
        JobManager(void);

        /** The viewing windows. */
        std::vector<std::shared_ptr<Job> > jobs;

        /** whether shutdown has been requested. do not start jobs again. */
        bool terminating;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_JOBMANAGER_H_INCLUDED */
