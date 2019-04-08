/*
 * vector_field_reader.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "vector_field_call.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Rectangle.h"

#include <array>
#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Reader for vector fields.
        *
        * @author Alexander Straub
        */
        class vector_field_reader : public core::Module
        {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static const char* ClassName() { return "vector_field_reader"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static const char* Description() { return "Reader to load vector fields from file"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static bool IsAvailable() { return true; }

            /**
            * Constructor
            */
            vector_field_reader();

            /**
            * Destructor
            */
            ~vector_field_reader();

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
                /** Bounding rectangle */
                vislib::math::Rectangle<float> bounding_rectangle;

                /** Grid resolution */
                std::array<unsigned int, 2> resolution;

                /** Grid positions */
                std::shared_ptr<std::vector<float>> positions;

                /** Vectors */
                std::shared_ptr<std::vector<float>> vectors;

                /** Current hash */
                SIZE_T hash;

            } stored_data;
        };
    }
}
