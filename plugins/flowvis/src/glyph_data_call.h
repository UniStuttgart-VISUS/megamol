/*
 * glyph_data_call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/math/Rectangle.h"

#include "Eigen/Dense"

#include <array>
#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Call for transporting a glyphs in an ready-to-use fashion (for OpenGL).
        *
        * @author Alexander Straub
        */
        class glyph_data_call : public core::AbstractGetDataCall
        {
        public:
            typedef core::factories::CallAutoDescription<glyph_data_call> glyph_data_description;

            /**
            * Human-readable class name
            */
            static const char* ClassName() { return "glyph_data_call"; }

            /**
            * Human-readable class description
            */
            static const char* Description() { return "Call transporting glyphs"; }

            /**
            * Number of available functions
            */
            static unsigned int FunctionCount() { return 2; }

            /**
            * Names of available functions
            */
            static const char* FunctionName(unsigned int idx)
            {
                switch (idx)
                {
                case 0: return "get_data";
                case 1: return "get_extent";
                }

                return nullptr;
            }

            /**
            * Constructor
            */
            glyph_data_call();

            /**
             * Getter for the bounding rectangle
             */
            const vislib::math::Rectangle<float>& get_bounding_rectangle() const;

            /**
            * Does a valid bounding rectangle exist?
            */
            bool has_bounding_rectangle() const;

            /**
             * Setter for the bounding rectangle
             */
            void set_bounding_rectangle(const vislib::math::Rectangle<float>& bounding_rectangle);

            /**
            * Getter for the vertices defining the point glyphs
            */
            std::shared_ptr<std::vector<float>> get_point_vertices() const;

            /**
            * Getter for the vertices defining the line glyphs
            */
            std::shared_ptr<std::vector<float>> get_line_vertices() const;

            /**
            * Getter for the indices defining the point glyphs
            */
            std::shared_ptr<std::vector<unsigned int>> get_point_indices() const;

            /**
            * Getter for the indices defining the line glyphs
            */
            std::shared_ptr<std::vector<unsigned int>> get_line_indices() const;

            /**
            * Getter for the values of the points
            */
            std::shared_ptr<std::vector<float>> get_point_values() const;

            /**
            * Getter for the values of the lines
            */
            std::shared_ptr<std::vector<float>> get_line_values() const;

            /**
            * Add a point
            *
            * @param point Point position
            * @param value Value stored at the point
            */
            void add_point(const Eigen::Vector2f& point, float value);

            /**
            * Add a line
            *
            * @param points Points of the line
            * @param value Value stored at the line
            */
            void add_line(const std::vector<Eigen::Vector2f>& points, float value);

            /**
            * Clear all, removing all points and lines
            */
            void clear();

        protected:
            /** Bounding rectangle */
            vislib::math::Rectangle<float> bounding_rectangle;
            bool bounding_rectangle_valid;

            /** Vertices and indices defining the glyphs */
            std::shared_ptr<std::vector<float>> point_vertices, line_vertices;
            std::shared_ptr<std::vector<unsigned int>> point_indices, line_indices;
            std::shared_ptr<std::vector<float>> point_values, line_values;
        };
    }
}
