#include "stdafx.h"
#include "triangle_mesh_renderer_3d.h"

#include "mesh_data_call.h"
#include "triangle_mesh_call.h"

#include "flowvis/shader.h"

#include "mesh/GPUMeshCollection.h"
#include "mesh/MeshCalls.h"

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/Call.h"
#include "mmcore/CallGeneric.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/utility/DataHash.h"

#include "vislib/sys/Log.h"

#include "glad/glad.h"

#include "glowl/VertexLayout.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        triangle_mesh_renderer_3d::triangle_mesh_renderer_3d()
            : triangle_mesh_slot("get_triangle_mesh", "Triangle mesh input")
            , mesh_data_slot("get_mesh_data", "Mesh data input")
            , data_set("data_set", "Data set used for coloring the triangles")
            , mask("mask", "Validity mask to selectively hide unwanted vertices or triangles")
            , mask_color("mask_color", "Color for invalid values")
            , wireframe("wireframe", "Render as wireframe instead of filling the triangles")
            , triangle_mesh_hash(-1)
            , triangle_mesh_changed(false)
            , mesh_data_hash(-1)
            , mesh_data_changed(false)
        {
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

        bool triangle_mesh_renderer_3d::get_input_data() {
            auto tmc_ptr = this->triangle_mesh_slot.CallAs<triangle_mesh_call>();
            auto mdc_ptr = this->mesh_data_slot.CallAs<mesh_data_call>();

            if (tmc_ptr == nullptr) {
                vislib::sys::Log::DefaultLog.WriteError("Triangle mesh input is not connected");

                return false;
            }

            auto& tmc = *tmc_ptr;

            if (!tmc(0)) {
                vislib::sys::Log::DefaultLog.WriteError("Error getting triangle mesh");

                return false;
            }

            if (tmc.DataHash() != this->triangle_mesh_hash) {
                this->render_data.vertices = tmc.get_vertices();
                this->render_data.indices = tmc.get_indices();

                this->triangle_mesh_hash = tmc.DataHash();
                this->triangle_mesh_changed = true;
            }

            if (mdc_ptr != nullptr && (*mdc_ptr)(0) && mdc_ptr->DataHash() != this->mesh_data_hash &&
                !mdc_ptr->get_data_sets().empty()) {

                this->render_data.values = mdc_ptr->get_data(mdc_ptr->get_data_sets()[0]);

                this->mesh_data_hash = mdc_ptr->DataHash();
                this->mesh_data_changed = true;
            } else {
                this->render_data.values = std::make_shared<mesh_data_call::data_set>();

                this->render_data.values->min_value = 0.0f;
                this->render_data.values->max_value = 1.0f;

                this->render_data.values->transfer_function = "";
                this->render_data.values->transfer_function_dirty = true;

                this->render_data.values->data =
                    std::make_shared<std::vector<GLfloat>>(this->render_data.vertices->size() / 3, 1.0f);
            }

            return true;
        }

        bool triangle_mesh_renderer_3d::get_input_extent() {
            auto tmc_ptr = this->triangle_mesh_slot.CallAs<triangle_mesh_call>();
            auto mdc_ptr = this->triangle_mesh_slot.CallAs<mesh_data_call>();

            if (tmc_ptr == nullptr) {
                vislib::sys::Log::DefaultLog.WriteError("Triangle mesh input is not connected");

                return false;
            }

            auto& tmc = *tmc_ptr;

            if (!tmc(1)) {
                vislib::sys::Log::DefaultLog.WriteError("Error getting extents for the triangle mesh");

                return false;
            }

            if (tmc.get_dimension() != triangle_mesh_call::dimension_t::THREE) {
                vislib::sys::Log::DefaultLog.WriteError("Input triangle mesh must be three-dimensional");

                return false;
            }

            this->bounding_box = tmc.get_bounding_box();

            return true;
        }

        bool triangle_mesh_renderer_3d::getDataCallback(core::Call& call) {
            auto& gmdc = static_cast<mesh::CallGPUMeshData&>(call);

            if (!get_input_data()) {
                return false;
            }

            if (this->triangle_mesh_changed || this->mesh_data_changed) {
                this->m_gpu_meshes = std::make_shared<mesh::GPUMeshCollection>();

                using vbi_t = typename std::vector<GLfloat>::iterator;
                using ibi_t = typename std::vector<GLuint>::iterator;

                std::vector<glowl::VertexLayout::Attribute> attributes{
                    glowl::VertexLayout::Attribute(3, GL_FLOAT, GL_FALSE, 0),
                    glowl::VertexLayout::Attribute(1, GL_FLOAT, GL_FALSE, 0)
                };

                glowl::VertexLayout vertex_descriptor(0, attributes);

                std::vector<std::pair<vbi_t, vbi_t>> vertex_buffer{
                    { this->render_data.vertices->begin(), this->render_data.vertices->end() },
                    { this->render_data.values->data->begin(), this->render_data.values->data->end() }};

                std::pair<ibi_t, ibi_t> index_buffer{ this->render_data.indices->begin(), this->render_data.indices->end() };

                this->m_gpu_meshes->template addMesh<vbi_t, ibi_t>(
                    vertex_descriptor, vertex_buffer, index_buffer, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);

                this->triangle_mesh_changed = false;
                this->mesh_data_changed = false;
            }

            gmdc.setData(this->m_gpu_meshes);

            return true;
        }

        bool triangle_mesh_renderer_3d::getMetaDataCallback(core::Call& call) {
            auto& gmdc = static_cast<mesh::CallGPUMeshData&>(call);

            if (!get_input_extent()) {
                return false;
            }

            core::BoundingBoxes_2 bbox;
            bbox.SetBoundingBox(this->bounding_box);
            bbox.SetClipBox(this->bounding_box);

            gmdc.setMetaData(core::Spatial3DMetaData{
                core::utility::DataHash(this->triangle_mesh_hash, this->mesh_data_hash), 1, 0, bbox});

            return true;
        }
    }
}
