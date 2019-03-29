/*
 * AbstractStreamProvider.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include <iostream>

namespace megamol {
namespace core {

    /**
    * Provides a stream.
    *
    * @author Alexander Straub
    */
    class AbstractStreamProvider : public Module {

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
        virtual std::ostream& GetStream() = 0;

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

    private:
        /**
         * Starts the job.
         *
         * @return true if the job has been successfully started.
         */
        bool Run(Call&);

        /** Input slot  */
        CallerSlot inputSlot;

        /** Tick slot */
        CalleeSlot tickSlot;
    };

}
}