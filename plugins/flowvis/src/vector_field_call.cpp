#include "stdafx.h"
#include "vector_field_call.h"

#include "vislib/math/Rectangle.h"

#include <array>
#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        vector_field_call::vector_field_call() : vectors(nullptr)
        {
            SetDataHash(-1);
        }

        const vislib::math::Rectangle<float>& vector_field_call::get_bounding_rectangle() const
        {
            return this->bounding_rectangle;
        }

        void vector_field_call::set_bounding_rectangle(const vislib::math::Rectangle<float>& bounding_rectangle)
        {
            this->bounding_rectangle = bounding_rectangle;
        }

        const std::array<unsigned int, 2>& vector_field_call::get_resolution() const
        {
            return this->resolution;
        }

        void vector_field_call::set_resolution(std::array<unsigned int, 2> resolution)
        {
            this->resolution = resolution;
        }

        std::shared_ptr<std::vector<float>> vector_field_call::get_positions() const
        {
            return this->positions;
        }

        void vector_field_call::set_positions(std::shared_ptr<std::vector<float>> positions)
        {
            this->positions = positions;
        }

        std::shared_ptr<std::vector<float>> vector_field_call::get_vectors() const
        {
            return this->vectors;
        }

        void vector_field_call::set_vectors(std::shared_ptr<std::vector<float>> vectors)
        {
            this->vectors = vectors;
        }
    }
}
