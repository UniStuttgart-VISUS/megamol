#include "stdafx.h"
#include "triangle_mesh_renderer_3d.h"

#include "mesh_data_call.h"
#include "triangle_mesh_call.h"

#include "flowvis/shader.h"

#include "mesh/MeshCalls.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/TransferFunctionParam.h"

#include "vislib/sys/Log.h"

#include "glad/glad.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        triangle_mesh_renderer_3d::triangle_mesh_renderer_3d() :
            render_task("render_task", "Render task for drawing"),
            gpu_mesh("gpu_mesh", "Information how the mesh is stored on the GPU"),
            triangle_mesh_slot("get_triangle_mesh", "Triangle mesh input"), triangle_mesh_hash(-1),
            mesh_data_slot("get_mesh_data", "Mesh data input"), mesh_data_hash(-1),
            data_set("data_set", "Data set used for coloring the triangles"),
            mask("mask", "Validity mask to selectively hide unwanted vertices or triangles"),
            mask_color("mask_color", "Color for invalid values"),
            wireframe("wireframe", "Render as wireframe instead of filling the triangles")
        {
            // Setup output
            this->render_task.SetCallback(mesh::CallGPURenderTaskData::ClassName(), "GetData", &triangle_mesh_renderer_3d::get_task_callback);
            this->render_task.SetCallback(mesh::CallGPURenderTaskData::ClassName(), "GetMetaData", &triangle_mesh_renderer_3d::get_task_extent_callback);
            this->MakeSlotAvailable(&this->render_task);

            this->gpu_mesh.SetCallback(mesh::CallGPUMeshData::ClassName(), "GetData", &triangle_mesh_renderer_3d::get_mesh_callback);
            this->gpu_mesh.SetCallback(mesh::CallGPUMeshData::ClassName(), "GetMetaData", &triangle_mesh_renderer_3d::get_mesh_extent_callback);
            this->MakeSlotAvailable(&this->gpu_mesh);

            // Connect input slots
            this->triangle_mesh_slot.SetCompatibleCall<triangle_mesh_call::triangle_mesh_description>();
            this->MakeSlotAvailable(&this->triangle_mesh_slot);

            this->mesh_data_slot.SetCompatibleCall<mesh_data_call::mesh_data_description>();
            this->MakeSlotAvailable(&this->mesh_data_slot);

            // Connect parameter slots
            this->data_set << new core::param::FlexEnumParam("");
            this->MakeSlotAvailable(&this->data_set);

            this->mask << new core::param::FlexEnumParam("");
            this->MakeSlotAvailable(&this->mask);

            this->mask_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
            this->MakeSlotAvailable(&this->mask_color);

            this->wireframe << new core::param::BoolParam(false);
            this->MakeSlotAvailable(&this->wireframe);
        }

        triangle_mesh_renderer_3d::~triangle_mesh_renderer_3d()
        {
            this->Release();
        }

        bool triangle_mesh_renderer_3d::create()
        {
            return true;
        }

        void triangle_mesh_renderer_3d::release()
        {
        }

        bool triangle_mesh_renderer_3d::get_task_callback(core::Call& caller) {


            return true;
        }

        bool triangle_mesh_renderer_3d::get_task_extent_callback(core::Call& caller) {


            return true;
        }

        bool triangle_mesh_renderer_3d::get_mesh_callback(core::Call& caller) {


            return true;
        }

        bool triangle_mesh_renderer_3d::get_mesh_extent_callback(core::Call& caller){


            return true;
        }
    }
}
