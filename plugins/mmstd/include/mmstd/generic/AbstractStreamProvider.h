/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CallerSlot.h"
#include "mmstd/job/AbstractTickJob.h"

#include <iostream>

namespace megamol::core {

/**
 * Provides a stream.
 *
 * @author Alexander Straub
 */
class AbstractStreamProvider : public job::AbstractTickJob {

public:
    /**
     * Constructor
     */
    AbstractStreamProvider();

    /**
     * Destructor
     */
    ~AbstractStreamProvider() override;

protected:
    /**
     * Callback function providing the stream.
     *
     * @return Stream
     */
    virtual std::iostream& GetStream() = 0;

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

    /**
     * Starts the job.
     *
     * @return true if the job has been successfully started.
     */
    bool run() override;

private:
    /** Input slot  */
    CallerSlot inputSlot;
};

} // namespace megamol::core
