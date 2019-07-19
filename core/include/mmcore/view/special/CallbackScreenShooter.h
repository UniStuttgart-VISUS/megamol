/*
 * CallbackScreenShooter.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractCallbackCall.h"
#include "mmcore/AbstractWriterParams.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/job/AbstractTickJob.h"
#include "mmcore/view/special/ScreenShooter.h"

#include <functional>

namespace megamol {
namespace core {
namespace view {
namespace special {

    /**
    * Call transporting a callback delivering an ostream object for writing data.
    *
    * @author Alexander Straub
    */
    class MEGAMOLCORE_API CallbackScreenShooterCall : public AbstractCallbackCall<std::function<void()>> {

    public:
        typedef factories::CallAutoDescription<CallbackScreenShooterCall> CallbackScreenShooterDescription;

        /**
        * Human-readable class name
        */
        static const char* ClassName() { return "CallbackScreenShooterCall"; }

        /**
        * Human-readable class description
        */
        static const char *Description() { return "Call transporting a callback for shooting screenshots"; }

        /**
        * Number of available functions
        */
        static unsigned int FunctionCount() { return 1; }

        /**
        * Names of available functions
        */
        static const char * FunctionName(unsigned int idx) {

            switch (idx)
            {
            case 0: return "SetCallback";
            }

            return nullptr;
        }
    };

    /**
     * Class implementing the screen shooter using a callback
     *
     * @author Alexander Straub
     */
    class MEGAMOLCORE_API CallbackScreenShooter : public ScreenShooter, protected AbstractWriterParams {

    public:
        using FunctionT = std::function<void()>;

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName() {
            return "CallbackScreenShooter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description() {
            return "A simple tick job module used to create large off-screen renderings";
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
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart() {
            return false;
        }

        /**
         * Ctor
         */
        CallbackScreenShooter();

        /**
         * Dtor
         */
        virtual ~CallbackScreenShooter();

    protected:
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

        /*
         * Create the screenshot.
         */
        void CreateScreenshot();

        /** Input slot */
        CallerSlot inputSlot;

        /** Tick slot */
        CalleeSlot tickSlot;
    };

}
}
}
}
