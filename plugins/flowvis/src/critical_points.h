/*
 * critical_points.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "Eigen/Dense"

#include <utility>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for calculating critical points of a vector field.
        *
        * @author Alexander Straub
        */
        class critical_points : public core::Module
        {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "critical_points"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "Calculate critical points of a 2D vector field"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            critical_points();

            /**
             * Finalises an instance.
             */
            virtual ~critical_points();

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
            /** Types of critical points */
            enum type
            {
                NONE = -1, UNHANDLED, POINT
            };

            /** Cell */
            struct cell_t
            {
                Eigen::Vector2f bottom_left, bottom_right, top_left, top_right;
                Eigen::Vector2f bottom_left_corner, top_right_corner;
            };

            /**
            * Extract a critical point from a cell, if it exists
            *
            * @param cell Cell in which to look for a critical point
            *
            * @return Critical point and its corresponding type
            */
            std::pair<type, Eigen::Vector2f> critical_points::extract_critical_point(const cell_t& cell) const;

            /**
            * Linear interpolate position based on value
            *
            * @param left "Left" position
            * @param right "Right" position
            * @param value_left Value at "left" position
            * @param value_right Value at "right" position
            *
            * @return Position at which the value is zero
            */
            Eigen::Vector2f linear_interpolate_position(const Eigen::Vector2f& left, const Eigen::Vector2f& right, float value_left, float value_right) const;

            /**
            * Linear interpolate the value at a given position
            *
            * @param left "Left" position
            * @param right "Right" position
            * @param value_left Value at "left" position
            * @param value_right Value at "right" position
            * @param position Position at which the value should be interpolated
            *
            * @return Interpolated value
            */
            float linear_interpolate_value(float left, float right, float value_left, float value_right, float position) const;

            /** Callbacks for the triangle mesh */
            bool get_glyph_data_callback(core::Call& call);
            bool get_glyph_extent_callback(core::Call& call);

            /** Output slot for the glyphs */
            core::CalleeSlot glyph_slot;
            SIZE_T glyph_hash;
            std::vector<Eigen::Vector2f> glyph_output;

            /** Input slot for getting an input vector field */
            core::CallerSlot vector_field_slot;
            SIZE_T vector_field_hash;

            /** Parameter for the boundary */
            core::param::ParamSlot boundary;
        };
    }
}