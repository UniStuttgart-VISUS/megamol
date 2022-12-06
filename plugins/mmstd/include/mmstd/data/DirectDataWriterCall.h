/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <functional>
#include <iostream>

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/generic/AbstractCallbackCall.h"

namespace megamol::core {

/**
 * Call transporting a callback delivering an ostream object for writing data.
 *
 * @author Alexander Straub
 */
class DirectDataWriterCall : public AbstractCallbackCall<std::function<std::ostream&()>> {

public:
    typedef factories::CallAutoDescription<DirectDataWriterCall> DirectDataWriterDescription;

    /**
     * Human-readable class name
     */
    static const char* ClassName() {
        return "DirectDataWriterCall";
    }

    /**
     * Human-readable class description
     */
    static const char* Description() {
        return "Call transporting a callback delivering an ostream object for writing data";
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
            return "SetCallback";
        }

        return nullptr;
    }
};

} // namespace megamol::core
