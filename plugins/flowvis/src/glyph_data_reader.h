/*
 * glyph_data_reader.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "glyph_data_call.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Reader for glyphs.
        *
        * @author Alexander Straub
        */
        class glyph_data_reader : public core::Module
        {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static const char* ClassName() { return "glyph_data_reader"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static const char* Description() { return "Reader to load glyphs from file"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static bool IsAvailable() { return true; }

            /**
            * Constructor
            */
            glyph_data_reader();

            /**
            * Destructor
            */
            ~glyph_data_reader();

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
             * Callbacks for reading data from file.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            bool get_data(core::Call& call);
            bool get_extent(core::Call& call);

            /** Output slot */
            core::CalleeSlot output_slot;

            /** File path parameter */
            core::param::ParamSlot file_path_slot;

            /** Output data */
            struct data_t
            {
                /** Points */
                std::vector<std::pair<float, Eigen::Vector2f>> points;

                /** Lines */
                std::vector<std::pair<float, std::vector<Eigen::Vector2f>>> lines;

                /** Current hash */
                SIZE_T hash;

            } stored_data;
        };
    }
}
