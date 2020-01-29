/*
 * line_strip.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh_data_call.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Rectangle.h"

#include "Eigen/Dense"

#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for connecting points into a line strip
        *
        * @author Alexander Straub
        */
        class line_strip : public core::Module {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "line_strip"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "Connect points into a line strip"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            line_strip();

            /**
             * Finalises an instance.
             */
            virtual ~line_strip();

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
            /** Get input data and extent from called modules */
            bool get_input_data();
            bool get_input_extent();

            /** Create lines using different methods */
            void create_lines_input_order(const std::vector<Eigen::Vector2f>& points);
            void create_lines_tsp(const std::vector<Eigen::Vector2f>& points);

            /** Callbacks for the computed seed lines */
            bool get_lines_data(core::Call& call);
            bool get_lines_extent(core::Call& call);

            /** Output slot for computed lines */
            core::CalleeSlot line_strip_slot;

            /** Input slot for getting the points */
            core::CallerSlot points_slot;

            /** Parameters for defining the method for connection */
            core::param::ParamSlot method;

            /** Bounding rectangle */
            vislib::math::Rectangle<float> bounding_rectangle;

            /** Input points */
            SIZE_T points_hash;
            bool points_changed;

            std::vector<std::pair<Eigen::Vector2f, float>> points;

            /** Output lines */
            SIZE_T line_strip_hash;

            std::pair<float, std::vector<Eigen::Vector2f>> lines;
        };
    }
}
