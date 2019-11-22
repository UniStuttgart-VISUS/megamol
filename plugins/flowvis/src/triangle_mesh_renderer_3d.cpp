#include "stdafx.h"
#include "triangle_mesh_renderer_3d.h"

#include "mesh_data_call.h"
#include "triangle_mesh_call.h"

#include "flowvis/shader.h"

#include "mesh/GPUMeshCollection.h"
#include "mesh/MeshCalls.h"

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/Call.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/utility/DataHash.h"

#include "vislib/sys/Log.h"

#include "glad/glad.h"

#include "glowl/VertexLayout.hpp"

#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace megamol {
namespace flowvis {

triangle_mesh_renderer_3d::triangle_mesh_renderer_3d()
    : triangle_mesh_slot("get_triangle_mesh", "Triangle mesh input")
    , mesh_data_slot("get_mesh_data", "Mesh data input")
    , data_set("data_set", "Data set used for coloring the triangles")
    , default_color("default_color", "Default color if no dataset is specified")
    , triangle_mesh_hash(-1)
    , triangle_mesh_changed(false)
    , mesh_data_hash(-1)
    , mesh_data_changed(false) {

    // Connect input slots
    this->triangle_mesh_slot.SetCompatibleCall<triangle_mesh_call::triangle_mesh_description>();
    this->MakeSlotAvailable(&this->triangle_mesh_slot);

    this->mesh_data_slot.SetCompatibleCall<mesh_data_call::mesh_data_description>();
    this->MakeSlotAvailable(&this->mesh_data_slot);

    // Connect parameter slots
    this->data_set << new core::param::FlexEnumParam("");
    this->MakeSlotAvailable(&this->data_set);

    this->default_color << new core::param::ColorParam(0.7f, 0.7f, 0.7f, 1.0f);
    this->MakeSlotAvailable(&this->default_color);

    // Disconnect inherited slots
    this->SetSlotUnavailable(&this->m_renderTask_rhs_slot);
    this->SetSlotUnavailable(&this->m_mesh_slot);
    this->SetSlotUnavailable(&this->m_light_slot);
}

triangle_mesh_renderer_3d::~triangle_mesh_renderer_3d() { this->Release(); }

bool triangle_mesh_renderer_3d::create() {
    mesh::AbstractGPURenderTaskDataSource::create();

    return true;
}

void triangle_mesh_renderer_3d::release() {}

bool triangle_mesh_renderer_3d::get_input_data() {
    auto tmc_ptr = this->triangle_mesh_slot.CallAs<triangle_mesh_call>();
    auto mdc_ptr = this->mesh_data_slot.CallAs<mesh_data_call>();
    auto gmd_ptr = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();

    if (tmc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Triangle mesh input is not connected");

        return false;
    }

    if (gmd_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("No shader connected");

        return false;
    }

    auto& tmc = *tmc_ptr;
    auto& gmd = *gmd_ptr;

    if (!tmc(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting triangle mesh");

        return false;
    }

    if (!gmd(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting shader");

        return false;
    }

    if (tmc.DataHash() != this->triangle_mesh_hash) {
        this->render_data.vertices = tmc.get_vertices();
        this->render_data.normals = tmc.get_normals();
        this->render_data.indices = tmc.get_indices();

        this->triangle_mesh_hash = tmc.DataHash();
        this->triangle_mesh_changed = true;
    }

    if (gmd.getData()->getMaterials().size() == 0) {
        vislib::sys::Log::DefaultLog.WriteError("No shader attached for mesh rendering");

        return false;
    }

    this->render_data.shader = gmd.getData()->getMaterials().front().shader_program;

    if (mdc_ptr != nullptr && (*mdc_ptr)(0) && !mdc_ptr->get_data_sets().empty() &&
        mdc_ptr->DataHash() != this->mesh_data_hash) {

        this->data_set.Param<core::param::FlexEnumParam>()->ClearValues();

        for (const auto& data_set_name : mdc_ptr->get_data_sets()) {
            this->data_set.Param<core::param::FlexEnumParam>()->AddValue(data_set_name);
        }

        this->mesh_data_hash = mdc_ptr->DataHash();
    }

    if (mdc_ptr != nullptr) {
        this->render_data.values = mdc_ptr->get_data(this->data_set.Param<core::param::FlexEnumParam>()->Value());

        if (this->render_data.values != nullptr &&
            (this->render_data.values->transfer_function_dirty || this->data_set.IsDirty())) {
            this->data_set.ResetDirty();
            this->render_data.values->transfer_function_dirty = false;
            this->mesh_data_changed = true;
        }
    }

    if (this->render_data.values == nullptr) {
        this->render_data.values = std::make_shared<mesh_data_call::data_set>();

        this->render_data.values->min_value = 0.0f;
        this->render_data.values->max_value = 1.0f;

        const auto color = this->default_color.Param<core::param::ColorParam>()->Value();

        std::stringstream ss;
        ss << "{\"Interpolation\":\"LINEAR\",\"Nodes\":["
           << "[" << color[0] << "," << color[1] << "," << color[2] << "," << color[3] << ",0.0,0.05000000074505806],"
           << "[" << color[0] << "," << color[1] << "," << color[2] << "," << color[3] << ",1.0,0.05000000074505806]]"
           << ",\"TextureSize\":2,\"ValueRange\":[0.0,1.0]}";

        this->render_data.values->transfer_function = ss.str();
        this->render_data.values->transfer_function_dirty = false;

        this->render_data.values->data =
            std::make_shared<std::vector<GLfloat>>(this->render_data.vertices->size() / 3, 1.0f);

        this->mesh_data_changed = true;
    }

    return true;
}

bool triangle_mesh_renderer_3d::get_input_extent() {
    auto tmc_ptr = this->triangle_mesh_slot.CallAs<triangle_mesh_call>();
    auto mdc_ptr = this->triangle_mesh_slot.CallAs<mesh_data_call>();
    auto gmd_ptr = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();

    if (tmc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Triangle mesh input is not connected");

        return false;
    }

    if (gmd_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("No shader connected");

        return false;
    }

    auto& tmc = *tmc_ptr;
    auto& gmd = *gmd_ptr;

    if (!tmc(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting extents for the triangle mesh");

        return false;
    }

    if (tmc.get_dimension() != triangle_mesh_call::dimension_t::THREE) {
        vislib::sys::Log::DefaultLog.WriteError("Input triangle mesh must be three-dimensional");

        return false;
    }

    if (!gmd(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting meta data for the shader");

        return false;
    }

    this->bounding_box = tmc.get_bounding_box();

    return true;
}

bool triangle_mesh_renderer_3d::getDataCallback(core::Call& call) {
    auto& grtc = static_cast<mesh::CallGPURenderTaskData&>(call);

    if (!get_input_data()) {
        return false;
    }

    // Set (empty) render task
    std::shared_ptr<mesh::GPURenderTaskCollection> rt_collection;

    if (grtc.getData() == nullptr) {
        rt_collection = this->m_gpu_render_tasks;
        grtc.setData(rt_collection);
    } else {
        rt_collection = grtc.getData();
    }

    if (this->triangle_mesh_changed || this->mesh_data_changed) {
        // Create mesh
        this->render_data.mesh = std::make_shared<mesh::GPUMeshCollection>();

        using vbi_t = typename std::vector<GLfloat>::iterator;
        using ibi_t = typename std::vector<GLuint>::iterator;

        std::vector<glowl::VertexLayout::Attribute> attributes{glowl::VertexLayout::Attribute(3, GL_FLOAT, GL_FALSE, 0),
            glowl::VertexLayout::Attribute(1, GL_FLOAT, GL_FALSE, 0)};

        if (this->render_data.normals != nullptr) {
            attributes.emplace_back(3, GL_FLOAT, GL_TRUE, 0);
        }

        glowl::VertexLayout vertex_descriptor(0, attributes);

        std::vector<std::pair<vbi_t, vbi_t>> vertex_buffer{
            {this->render_data.vertices->begin(), this->render_data.vertices->end()},
            {this->render_data.values->data->begin(), this->render_data.values->data->end()}};

        if (this->render_data.normals != nullptr) {
            vertex_buffer.push_back({this->render_data.normals->begin(), this->render_data.normals->end()});
        }

        std::pair<ibi_t, ibi_t> index_buffer{this->render_data.indices->begin(), this->render_data.indices->end()};

        this->render_data.mesh->template addMesh<vbi_t, ibi_t>(
            vertex_descriptor, vertex_buffer, index_buffer, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);

        // Create texture for transfer function
        std::vector<GLfloat> texture_data;
        std::array<float, 2> _unused__texture_range;
        UINT transfer_function_size;

        core::param::TransferFunctionParam::TransferFunctionTexture(
            this->render_data.values->transfer_function, texture_data, transfer_function_size, _unused__texture_range);

        if (this->render_data.transfer_function != 0) {
            glDeleteTextures(1, &this->render_data.transfer_function);
        }

        glGenTextures(1, &this->render_data.transfer_function);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, this->render_data.transfer_function);

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, static_cast<GLsizei>(transfer_function_size), 0, GL_RGBA, GL_FLOAT,
            static_cast<GLvoid*>(texture_data.data()));

        glBindTexture(GL_TEXTURE_1D, 0);

        const auto transfer_function_handle = glGetTextureHandleARB(this->render_data.transfer_function);
        glMakeTextureHandleResidentARB(transfer_function_handle);

        // Create render task
        auto mesh_data = this->render_data.mesh->getSubMeshData().front();
        auto mesh = this->render_data.mesh->getMeshes()[mesh_data.batch_index].mesh;

        std::vector<glowl::DrawElementsCommand> draw_commands(1, mesh_data.sub_mesh_draw_command);

        std::array<uint8_t, sizeof(GLuint64) + 2 * sizeof(float)> per_draw_data;
        std::size_t offset = 0;
        std::memcpy(&per_draw_data[offset], &transfer_function_handle, sizeof(GLuint64));
        std::memcpy(&per_draw_data[offset += sizeof(GLuint64)], &this->render_data.values->min_value, sizeof(float));
        std::memcpy(&per_draw_data[offset += sizeof(float)], &this->render_data.values->max_value, sizeof(float));

        rt_collection->clear();
        rt_collection->addRenderTasks(this->render_data.shader, mesh, draw_commands, per_draw_data);

        this->triangle_mesh_changed = false;
        this->mesh_data_changed = false;
    }

    return true;
}

bool triangle_mesh_renderer_3d::getMetaDataCallback(core::Call& call) {
    auto& grtc = static_cast<mesh::CallGPURenderTaskData&>(call);

    if (!get_input_extent()) {
        return false;
    }

    core::BoundingBoxes_2 bbox;
    bbox.SetBoundingBox(this->bounding_box);
    bbox.SetClipBox(this->bounding_box);

    grtc.setMetaData(
        core::Spatial3DMetaData{core::utility::DataHash(this->triangle_mesh_hash, this->mesh_data_hash), 1, 0, bbox});

    return true;
}

} // namespace flowvis
} // namespace megamol
