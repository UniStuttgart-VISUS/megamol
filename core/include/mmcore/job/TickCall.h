/*
 * TickCall.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace core {
namespace job {

    /**
    * Call for propagating a tick.
    *
    * @author Alexander Straub
    */
    class MEGAMOLCORE_API TickCall : public core::AbstractGetDataCall
    {
    public:
        typedef core::factories::CallAutoDescription<TickCall> TickCallDescription;

        /**
        * Human-readable class name
        */
        static const char* ClassName() { return "TickCall"; }

        /**
        * Human-readable class description
        */
        static const char *Description() { return "Call for propagating a tick"; }

        /**
        * Number of available functions
        */
        static unsigned int FunctionCount() { return 1; }

        /**
        * Names of available functions
        */
        static const char * FunctionName(unsigned int idx)
        {
            switch (idx)
            {
            case 0: return "tick";
            }

            return nullptr;
        }
    };

}
}
}