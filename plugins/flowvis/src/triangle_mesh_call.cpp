#include "stdafx.h"
#include "triangle_mesh_call.h"

#include "vislib/math/Rectangle.h"

#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        const vislib::math::Rectangle<float>& triangle_mesh_call::get_bounding_rectangle() const
        {
            return this->bounding_rectangle;
        }

        void triangle_mesh_call::set_bounding_rectangle(const vislib::math::Rectangle<float>& bounding_rectangle)
        {
            this->bounding_rectangle = bounding_rectangle;
        }

        std::shared_ptr<std::vector<float>> triangle_mesh_call::get_vertices() const
        {
            return this->vertices;
        }

        void triangle_mesh_call::set_vertices(std::shared_ptr<std::vector<float>> vertices)
        {
            this->vertices = vertices;
        }

        std::shared_ptr<std::vector<unsigned int>> triangle_mesh_call::get_indices() const
        {
            return this->indices;
        }

        void triangle_mesh_call::set_indices(std::shared_ptr<std::vector<unsigned int>> indices)
        {
            this->indices = indices;
        }
    }
}