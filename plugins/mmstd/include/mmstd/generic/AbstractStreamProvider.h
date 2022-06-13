/*
 * AbstractStreamProvider.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmstd/job/AbstractTickJob.h"

#include <iostream>

namespace megamol {
namespace core {

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
    ~AbstractStreamProvider();

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
    virtual bool create() override;

    /**
     * Implementation of 'Release'.
     */
    virtual void release() override;

    /**
     * Starts the job.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool run() override;

private:
    /** Input slot  */
    CallerSlot inputSlot;
};

} // namespace core
} // namespace megamol
