/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/job/AbstractThreadedJob.h"

namespace megamol::core::job {

/**
 * Class implementing a simple thread for the job.
 */
class JobThread : public AbstractThreadedJob, public Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "JobThread";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Job thread controller";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Ctor
     */
    JobThread();

    /**
     * Dtor
     */
    ~JobThread() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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
    DWORD Run(void* userData) override;
};

} // namespace megamol::core::job
