/*
 * AbstractDataWriterCall.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace core {

    /**
    * Call transporting a callback for writing data.
    *
    * @author Alexander Straub
    */
    template <typename FunctionT>
    class AbstractDataWriterCall : public Call {

    public:
        using AbstractDataWriterDescription = factories::CallAutoDescription<AbstractDataWriterCall<FunctionT>>;

        /**
        * Human-readable class name
        */
        static const char* ClassName() { return "AbstractDataWriterDescription"; }

        /**
        * Human-readable class description
        */
        static const char *Description() { return "Call transporting a callback for writing data"; }

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

        /**
        * Constructor
        */
        AbstractDataWriterCall() {}

        /**
        * Set the callback
        *
        * @param Callback New callback
        */
        void SetCallback(FunctionT callback)
        {
            this->callback = callback;
        }

        /**
        * Get the stored callback
        *
        * @return Callback
        */
        FunctionT GetCallback() const
        {
            return this->callback;
        }

    private:
        /** Store callback */
        FunctionT callback;
    };

}
}