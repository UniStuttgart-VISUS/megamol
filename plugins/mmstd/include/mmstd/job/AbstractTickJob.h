/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

namespace megamol::core::job {

/**
 * Module for propagating a tick.
 *
 * @author Alexander Straub
 */
class AbstractTickJob : public core::Module {

public:
    /**
     * Constructor
     */
    AbstractTickJob();

    /**
     * Destructor
     */
    ~AbstractTickJob() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override = 0;

    /**
     * Implementation of 'Release'.
     */
    void release() override = 0;

    /**
     * Run this job.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool run() = 0;

private:
    /**
     * Starts the job.
     *
     * @return true if the job has been successfully started.
     */
    bool Run(Call&);

    /** Tick slot */
    CalleeSlot tickSlot;
};

} // namespace megamol::core::job
