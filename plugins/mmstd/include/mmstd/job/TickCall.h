/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetDataCall.h"

namespace megamol::core::job {

/**
 * Call for propagating a tick.
 *
 * @author Alexander Straub
 */
class TickCall : public core::AbstractGetDataCall {
public:
    typedef core::factories::CallAutoDescription<TickCall> TickCallDescription;

    /**
     * Human-readable class name
     */
    static const char* ClassName() {
        return "TickCall";
    }

    /**
     * Human-readable class description
     */
    static const char* Description() {
        return "Call for propagating a tick";
    }

    /**
     * Number of available functions
     */
    static unsigned int FunctionCount() {
        return 1;
    }

    /**
     * Names of available functions
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "tick";
        }

        return nullptr;
    }
};

} // namespace megamol::core::job
