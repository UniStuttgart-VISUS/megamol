/*
 * implicit_topology_call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "implicit_topology_results.h"

#include "mmcore/AbstractCallbackCall.h"

#include <functional>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Call for providing a callback for the results writer.
        *
        * @author Alexander Straub
        */
        class implicit_topology_writer_call : public core::AbstractCallbackCall<std::function<bool(const implicit_topology_results&)>>
        {
        public:
            typedef core::factories::CallAutoDescription<implicit_topology_writer_call> implicit_topology_writer_description;

            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static const char* ClassName() { return "implicit_topology_writer_call"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static const char* Description() { return "Call to provide a callback to save results from implicit topology computation to file"; }

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
        * Call for providing a callback for the results reader.
        *
        * @author Alexander Straub
        */
        class implicit_topology_reader_call : public core::AbstractCallbackCall<std::function<bool(implicit_topology_results&)>>
        {
        public:
            typedef core::factories::CallAutoDescription<implicit_topology_reader_call> implicit_topology_reader_description;

            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static const char* ClassName() { return "implicit_topology_reader_call"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static const char* Description() { return "Call to provide a callback to load results from implicit topology computation from file"; }

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
