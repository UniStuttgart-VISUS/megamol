#include "stdafx.h"
#include "implicit_topology.h"

#include "mmcore/Call.h"

#include "vislib/math/Rectangle.h"

#include "mesh_data_call.h"
#include "triangle_mesh_call.h"

#include "glad/glad.h"

namespace megamol
{
    namespace flowvis
    {
        implicit_topology::implicit_topology() :
            triangle_mesh_slot("set_triangle_mesh", "Triangle mesh output"),
            mesh_data_slot("set_mesh_data", "Mesh data output")
        {
            // Connect output
            this->triangle_mesh_slot.SetCallback(triangle_mesh_call::ClassName(), triangle_mesh_call::FunctionName(0), &implicit_topology::get_triangle_data_callback);
            this->triangle_mesh_slot.SetCallback(triangle_mesh_call::ClassName(), triangle_mesh_call::FunctionName(1), &implicit_topology::get_triangle_extent_callback);
            this->MakeSlotAvailable(&this->triangle_mesh_slot);

            this->mesh_data_slot.SetCallback(mesh_data_call::ClassName(), mesh_data_call::FunctionName(0), &implicit_topology::get_data_data_callback);
            this->mesh_data_slot.SetCallback(mesh_data_call::ClassName(), mesh_data_call::FunctionName(1), &implicit_topology::get_data_extent_callback);
            this->MakeSlotAvailable(&this->mesh_data_slot);
        }

        implicit_topology::~implicit_topology()
        {
            this->Release();
        }

        bool implicit_topology::create()
        {
            return true;
        }

        void implicit_topology::release()
        {
        }

        bool implicit_topology::get_triangle_data_callback(core::Call& call)
        {
            auto* triangle_call = dynamic_cast<triangle_mesh_call*>(&call);
            if (triangle_call == nullptr) return false;

            triangle_call->set_vertices(std::make_shared<std::vector<GLfloat>>(std::initializer_list<GLfloat> {
                0.0f, 0.0f,
                1.0f, 0.0f,
                0.5f, 1.0f
            })); // TODO

            triangle_call->set_indices(std::make_shared<std::vector<GLuint>>(std::initializer_list<GLuint> {
                0, 1, 2
            })); // TODO

            triangle_call->SetDataHash(1); // TODO
            
            return true;
        }

        bool implicit_topology::get_triangle_extent_callback(core::Call& call)
        {
            auto* triangle_call = dynamic_cast<triangle_mesh_call*>(&call);
            if (triangle_call == nullptr) return false;

            triangle_call->set_bounding_rectangle(vislib::math::Rectangle<float>(0.0f, 0.0f, 1.0f, 1.0f));

            triangle_call->SetDataHash(1); // TODO

            return true;
        }

        bool implicit_topology::get_data_data_callback(core::Call& call)
        {
            auto* data_call = dynamic_cast<mesh_data_call*>(&call);
            if (data_call == nullptr) return false;

            data_call->set_data("color", std::make_shared<std::vector<GLfloat>>(std::initializer_list<GLfloat> {
                1.0f, 0.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 0.0f, 1.0f, 1.0f
            })); // TODO

            return true;
        }

        bool implicit_topology::get_data_extent_callback(core::Call& call)
        {
            auto* data_call = dynamic_cast<mesh_data_call*>(&call);
            if (data_call == nullptr) return false;

            data_call->set_data("color"); // TODO

            data_call->SetDataHash(1); // TODO

            return true;
        }

    }
}
