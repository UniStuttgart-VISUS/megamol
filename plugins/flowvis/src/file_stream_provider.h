/*
 * file_stream_provider.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "abstract_stream_provider.h"
#include "direct_data_writer_call.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/sys/Log.h"

#include <iostream>
#include <fstream>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Provides a stream.
        *
        * @author Alexander Straub
        */
        class file_stream_provider : public abstract_stream_provider
        {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static const char* ClassName() { return "file_stream_provider"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static const char* Description() { return "Provides a file stream"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static bool IsAvailable() { return true; }

            /**
             * Disallow usage in quickstarts.
             *
             * @return false
             */
            static bool SupportQuickstart() { return false; }

            /**
            * Constructor
            */
            file_stream_provider() : file_path("file_path", "Output file path")
            {
                this->file_path << new core::param::FilePathParam("");
                this->MakeSlotAvailable(&this->file_path);
            }

        protected:
            /**
            * Callback function providing the stream.
            *
            * @return Stream
            */
            virtual std::ostream& get_stream() override
            {
                if (!this->stream.is_open())
                {
                    // Open file for writing
                    this->stream.open(this->file_path.Param<core::param::FilePathParam>()->Value(), std::ios_base::out | std::ios_base::binary);

                    if (!this->stream.good())
                    {
                        vislib::sys::Log::DefaultLog.WriteWarn("Unable to open file '%s' for writing!",
                            this->file_path.Param<core::param::FilePathParam>()->Value());
                    }
                }

                return this->stream;
            }

        private:
            /** File path parameter */
            core::param::ParamSlot file_path;

            /** File stream */
            std::fstream stream;
        };
    }
}