/*
 * DirectDataWriterCall.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractDataWriterCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include <functional>
#include <iostream>

namespace megamol {
namespace core {

    /**
    * Call transporting a callback delivering an ostream object for writing data.
    *
    * @author Alexander Straub
    */
    class DirectDataWriterCall : public AbstractDataWriterCall<std::function<std::ostream&()>>
    {
    public:
        typedef factories::CallAutoDescription<DirectDataWriterCall> direct_data_writer_description;

        /**
        * Human-readable class name
        */
        static const char* ClassName() { return "DirectDataWriterCall"; }

        /**
        * Human-readable class description
        */
        static const char *Description() { return "Call transporting a callback delivering an ostream object for writing data"; }

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

}
}