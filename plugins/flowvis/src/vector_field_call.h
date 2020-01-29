/*
 * vector_field_call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/math/Rectangle.h"

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
        class vector_field_call : public core::AbstractGetDataCall
        {
        public:
            typedef core::factories::CallAutoDescription<vector_field_call> vector_field_description;

            /**
            * Human-readable class name
            */
            static const char* ClassName() { return "vector_field_call"; }

            /**
            * Human-readable class description
            */
            static const char* Description() { return "Call transporting a vector field"; }

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
            vector_field_call();

            /**
             * Getter for the bounding rectangle
             */
            const vislib::math::Rectangle<float>& get_bounding_rectangle() const;

            /**
             * Setter for the bounding rectangle
             */
            void set_bounding_rectangle(const vislib::math::Rectangle<float>& bounding_rectangle);

            /**
            * Getter for the grid resolution
            */
            const std::array<unsigned int, 2>& get_resolution() const;

            /**
            * Setter for the grid resolution
            */
            void set_resolution(std::array<unsigned int, 2> resolution);

            /**
            * Getter for the positions
            */
            std::shared_ptr<std::vector<float>> get_positions() const;

            /**
            * Setter for the positions
            */
            void set_positions(std::shared_ptr<std::vector<float>> positions);

            /**
            * Getter for the vectors
            */
            std::shared_ptr<std::vector<float>> get_vectors() const;

            /**
            * Setter for the vectors
            */
            void set_vectors(std::shared_ptr<std::vector<float>> vectors);

        protected:
            /** Bounding rectangle */
            vislib::math::Rectangle<float> bounding_rectangle;

            /** Grid resolution */
            std::array<unsigned int, 2> resolution;

            /** Grid positions */
            std::shared_ptr<std::vector<float>> positions;
            
            /** Vectors */
            std::shared_ptr<std::vector<float>> vectors;
        };
    }
}
