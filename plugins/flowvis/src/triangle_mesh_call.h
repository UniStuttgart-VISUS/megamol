#pragma once

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/math/Rectangle.h"

#include "glad/glad.h"

#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        class triangle_mesh_call : public core::AbstractGetDataCall
        {
        public:
            typedef core::factories::CallAutoDescription<triangle_mesh_call> triangle_mesh_description;

            /**
            * Human-readable class name
            */
            static const char* ClassName() { return "triangle_mesh_call"; }

            /**
            * Human-readable class description
            */
            static const char* Description() { return "Call transporting triangle data"; }

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
             * Getter for the bounding rectangle
             */
            const vislib::math::Rectangle<float>& get_bounding_rectangle() const;

            /**
             * Setter for the bounding rectangle
             */
            void set_bounding_rectangle(const vislib::math::Rectangle<float>& bounding_rectangle);

            /**
            * Getter for the vertices defining the triangle mesh
            */
            std::shared_ptr<std::vector<GLfloat>> get_vertices() const;

            /**
            * Setter for the vertices defining the triangle mesh
            */
            void set_vertices(std::shared_ptr<std::vector<GLfloat>> vertices);

            /**
            * Getter for the indices defining the triangle mesh
            */
            std::shared_ptr<std::vector<GLuint>> get_indices() const;

            /**
            * Setter for the indices defining the triangle mesh
            */
            void set_indices(std::shared_ptr<std::vector<GLuint>> indices);

        protected:
            /** Bounding rectangle */
            vislib::math::Rectangle<float> bounding_rectangle;

            /** Vertices and indices defining the triangle mesh */
            std::shared_ptr<std::vector<GLfloat>> vertices;
            std::shared_ptr<std::vector<GLuint>> indices;
        };
    }
}