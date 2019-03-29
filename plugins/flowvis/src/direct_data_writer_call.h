/*
 * direct_data_writer_call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "abstract_data_writer_call.h"

#include "mmcore/factories/CallAutoDescription.h"

#include <functional>
#include <iostream>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Call transporting a callback delivering an ostream object for writing data.
        *
        * @author Alexander Straub
        */
        class direct_data_writer_call : public abstract_data_writer_call<std::function<std::ostream&()>>
        {
        public:
            typedef core::factories::CallAutoDescription<direct_data_writer_call> direct_data_writer_description;

            /**
            * Human-readable class name
            */
            static const char* ClassName() { return "direct_data_writer_call"; }

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
            static const char * FunctionName(unsigned int idx)
            {
                switch (idx)
                {
                case 0: return "set_callback";
                }

                return nullptr;
            }
        };
    }
}