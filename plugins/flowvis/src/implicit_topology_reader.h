/*
 * implicit_topology_reader.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "implicit_topology_call.h"
#include "implicit_topology_results.h"

#include "mmcore/AbstractCallbackReader.h"

namespace megamol
{
    namespace flowvis
    {
        /**
        * Reader for results from implicit topology computation.
        *
        * @author Alexander Straub
        */
        class implicit_topology_reader : public core::AbstractCallbackReader<implicit_topology_reader_call, implicit_topology_results&>
        {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static const char* ClassName() { return "implicit_topology_reader"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static const char* Description() { return "Reader to load previous results from implicit topology computation from file"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static bool IsAvailable() { return true; }

            /**
            * Constructor
            */
            implicit_topology_reader();

            /**
            * Destructor
            */
            ~implicit_topology_reader();

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

            /**
            * Run this job.
            *
            * @return true if the job has been successfully started.
            */
            virtual bool read(const std::string& filename, implicit_topology_results& content) override;
        };
    }
}
