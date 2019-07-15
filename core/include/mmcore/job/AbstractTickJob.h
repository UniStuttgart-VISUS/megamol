/*
 * AbstractTickJob.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"

namespace megamol {
namespace core {
namespace job {

    /**
    * Module for propagating a tick.
    *
    * @author Alexander Straub
    */
    class MEGAMOLCORE_API AbstractTickJob : public core::Module {

    public:
        /**
        * Constructor
        */
        AbstractTickJob();

        /**
        * Destructor
        */
        virtual ~AbstractTickJob();

    protected:
        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create() = 0;

        /**
         * Implementation of 'Release'.
         */
        virtual void release() = 0;

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

}
}
}
