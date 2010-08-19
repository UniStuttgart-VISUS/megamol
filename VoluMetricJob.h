/*
 * VoluMetricJob.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VOLUMETRICJOB_H_INCLUDED
#define MEGAMOLCORE_VOLUMETRICJOB_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "job/AbstractThreadedJob.h"
#include "Module.h"
//#include "view/Renderer3DModule.h"
//#include "Call.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
//#include "vislib/Cuboid.h"
//#include "vislib/memutils.h"


namespace megamol {
namespace trisoup {

    /**
     * TODO: Document
     */
    class VoluMetricJob : public core::job::AbstractThreadedJob, public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "VoluMetricJob";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Measures stuff from something and Guido will write a better help text here!";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        VoluMetricJob(void);

        virtual ~VoluMetricJob(void);

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

        /**
         * Perform the work of a thread.
         *
         * @param userData A pointer to user data that are passed to the thread,
         *                 if it started.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *userData);

    private:

        core::CallerSlot getDataSlot;

        core::param::ParamSlot resultFilenameSlot;

    };

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VOLUMETRICJOB_H_INCLUDED */
